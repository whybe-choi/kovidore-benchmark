#!/usr/bin/env python3
"""
Generate word-level interpretability maps for vision document retrieval models.

This script visualizes similarity maps between query text and document images.
It supports multiple models (ColQwen2.5, ColPali, Jina Embeddings v4)
and generates individual similarity maps for each word in the query.

Usage:
    python examples/interpretability/generate_interpretability_maps.py --model_name vidore/colqwen2.5-v0.2 --task vqa
    python examples/interpretability/generate_interpretability_maps.py --model_name vidore/colpali-v1.3 --task mir --batch_size 16
"""

import argparse
import logging
from pathlib import Path

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, ColPali, ColPaliProcessor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

ALL_TASKS = ["vqa", "mir", "slide", "office", "finocr"]
AVAILABLE_MODELS = ["vidore/colqwen2.5-v0.2", "vidore/colpali-v1.3", "jinaai/jina-embeddings-v4"]


def map_tokens_to_words(query_text, display_tokens):
    """
    Utility function for visualizing similarity maps at the word level for Korean text.
    Maps tokenized outputs back to word boundaries by accumulating tokens until they match
    the original word length.

    Args:
        query_text: Original query text string
        display_tokens: List of decoded token strings from the tokenizer

    Returns:
        List of dictionaries containing word and corresponding token indices
    """
    words = query_text.split()
    word_groups = []
    token_idx = 0

    for word in words:
        if token_idx >= len(display_tokens):
            logger.warning(f"Ran out of tokens while processing word '{word}'")
            break

        word_token_indices = []
        accumulated = ""

        word_norm = word.replace(" ", "")

        while token_idx < len(display_tokens):
            current_token = display_tokens[token_idx]

            word_token_indices.append(token_idx)
            accumulated += current_token
            token_idx += 1

            accumulated_norm = accumulated.replace(" ", "")

            if len(accumulated_norm) >= len(word_norm):
                break

        if word_token_indices:
            word_groups.append(
                {
                    "word": word,
                    "token_indices": word_token_indices,
                }
            )

    return word_groups


def main(args):
    # =============================================
    # 1. Load Dataset
    # =============================================
    logger.info("Loading dataset...")
    task = args.task

    queries_df = pd.read_csv(f"data/{task}/queries.csv")
    corpus_df = pd.read_csv(f"data/{task}/corpus.csv")
    qrels_df = pd.read_csv(f"data/{task}/qrels.csv")

    examples = []
    for _, row in qrels_df.iterrows():
        qid = row["query-id"]
        cid = row["corpus-id"]

        query_row = queries_df[queries_df["query-id"] == qid]
        corpus_row = corpus_df[corpus_df["corpus-id"] == cid]

        if not query_row.empty and not corpus_row.empty:
            examples.append(
                {
                    "query_id": qid,
                    "corpus_id": cid,
                    "query": query_row["text"].values[0],
                    "image_path": corpus_row["image_path"].values[0],
                }
            )

    # =============================================
    # 2. Load Model and Processor
    # =============================================
    logger.info("Loading model and processor...")
    if "colqwen2.5" in args.model_name:
        processor = ColQwen2_5_Processor.from_pretrained(args.model_name)
        model = ColQwen2_5.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    elif "colpali" in args.model_name:
        processor = ColPaliProcessor.from_pretrained(args.model_name)
        model = ColPali.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    elif "jina-embeddings-v4" in args.model_name:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    model.eval()

    # =============================================
    # 3. Computing Embeddings
    # =============================================
    for batch_start in tqdm(range(0, len(examples), args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(examples))
        batch_examples = examples[batch_start:batch_end]

        images = [Image.open(example["image_path"]) for example in batch_examples]
        queries = [example["query"] for example in batch_examples]

        batch_images = processor.process_images(images=images).to(model.device)
        batch_queries = processor.process_texts(texts=queries).to(model.device)

        # Compute embeddings
        with torch.no_grad():
            if "jina-embeddings-v4" in args.model_name:
                image_output = model(task_label="retrieval", **batch_images)
                query_output = model(task_label="retrieval", **batch_queries)
                image_embeddings = image_output.multi_vec_emb
                query_embeddings = query_output.multi_vec_emb
            else:
                image_embeddings = model(**batch_images)
                query_embeddings = model(**batch_queries)

        # =============================================
        # 4. Generate Similarity Maps
        # =============================================
        batched_n_patches = []
        for image in images:
            if "jina-embeddings-v4" in args.model_name:
                patch_size = processor.image_processor.patch_size
                merge_size = processor.image_processor.merge_size
                spatial_merge_size = model.config.vision_config.spatial_merge_size
                height_new, width_new = smart_resize(
                    width=image.size[0],
                    height=image.size[1],
                    factor=patch_size * merge_size,
                    min_pixels=processor.image_processor.min_pixels,
                    max_pixels=processor.image_processor.max_pixels,
                )
                n_patches_x = width_new // patch_size // spatial_merge_size
                n_patches_y = height_new // patch_size // spatial_merge_size
                n_patches = (n_patches_x, n_patches_y)
            elif "colpali" in args.model_name:
                n_patches = processor.get_n_patches(
                    image_size=image.size,
                    patch_size=model.patch_size,
                )
            elif "colqwen2.5" in args.model_name:
                n_patches = processor.get_n_patches(
                    image_size=image.size,
                    spatial_merge_size=model.spatial_merge_size,
                )
            batched_n_patches.append(n_patches)
        image_mask = batch_images.input_ids == processor.image_token_id

        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=batched_n_patches,
            image_mask=image_mask,
        )

        # =============================================
        # 5. Process Each Sample
        # =============================================
        for idx, example in enumerate(batch_examples):
            logger.info(f"[{batch_start + idx + 1}/{len(examples)}] Query ID: {example['query_id']}")

            similarity_maps = batched_similarity_maps[idx]
            input_ids = batch_queries.input_ids[idx]
            query_text = example["query"]

            # Filter special tokens
            special_token_ids = set(processor.tokenizer.all_special_ids or [])

            filtered_tokens = []
            filtered_indices = []
            for i, token_id in enumerate(input_ids):
                tid = token_id.item()
                if tid not in special_token_ids:
                    filtered_tokens.append(
                        processor.tokenizer.decode([token_id], skip_special_tokens=False, errors="replace")
                    )
                    filtered_indices.append(i)

            # Skip "Query:" prefix
            query_start_idx = 0
            for i in range(len(filtered_tokens) - 1):
                if filtered_tokens[i] == "Query" and filtered_tokens[i + 1] == ":":
                    query_start_idx = i + 2
                    break

            filtered_tokens = filtered_tokens[query_start_idx:]
            filtered_indices = filtered_indices[query_start_idx:]
            filtered_sim_maps = similarity_maps[filtered_indices]

            # Clean tokens for display (remove special markers)
            display_tokens = [t.replace("Ġ", " ").replace("▁", " ") for t in filtered_tokens]

            logger.info(f"Query: '{query_text}'")
            logger.info(f"Tokens: {display_tokens}")

            word_groups = map_tokens_to_words(query_text, display_tokens)
            logger.info(f"{len(word_groups)} words: {[wg['word'] for wg in word_groups]}")

            output_dir = Path(args.output_dir) / args.model_name.split("/")[-1] / task / str(example["query_id"])
            output_dir.mkdir(parents=True, exist_ok=True)

            # =============================================
            # 6. Save Full Query Similarity Map
            # =============================================
            full_query_map = filtered_sim_maps.sum(dim=0)

            fig, ax = plot_similarity_map(
                image=images[idx],
                similarity_map=full_query_map,
                figsize=(8, 8),
            )
            ax.set_title(f"Query: {query_text}", fontsize=12)
            fig.savefig(output_dir / "00_query.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            # =============================================
            # 7. Save Word-level Similarity Maps
            # =============================================
            for word_idx, wg in enumerate(word_groups):
                word = wg["word"]
                token_indices = wg["token_indices"]

                # Sum similarity maps for all tokens in this word
                word_sim_map = filtered_sim_maps[token_indices].sum(dim=0)

                fig, ax = plot_similarity_map(
                    image=images[idx],
                    similarity_map=word_sim_map,
                    figsize=(8, 8),
                )
                ax.set_title(f"Word: '{word}'", fontsize=12)

                # Sanitize filename (remove special characters)
                safe_word = word.replace("/", "_").replace("\\", "_").replace("?", "")
                fig.savefig(output_dir / f"{word_idx + 1:02d}_{safe_word}.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

            logger.info(f"Saved {len(word_groups) + 1} similarity maps to {output_dir}")

        # Clear GPU memory
        del batch_images, batch_queries, image_embeddings, query_embeddings, batched_similarity_maps
        torch.cuda.empty_cache()

    logger.info("All tasks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="vidore/colqwen2.5-v0.2",
        choices=AVAILABLE_MODELS,
        help=f"Model name to evaluate (default: vidore/colqwen2.5-v0.2). Available: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vqa",
        choices=ALL_TASKS,
        help=f"Task to run (default: vqa). Available: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/interpretability",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    args = parser.parse_args()

    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task: {args.task}")
    logger.info("=" * 60)

    main(args)
