import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from datasets.video_dataset import CONCEPT_PROMPTS, CONCEPT_PROMPT_TEMPLATES


def main():
    parser = argparse.ArgumentParser(description="Build CLIP text embeddings for the 96 paper concepts.")
    parser.add_argument("--output", default="concept_text_embeddings.pt")
    parser.add_argument("--backend", choices=["transformers", "open_clip"], default="transformers")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    concepts = []
    for family in ["action", "object", "scene", "dynamic"]:
        concepts.extend(CONCEPT_PROMPTS[family])
    if len(concepts) != 96:
        raise ValueError(f"Expected 96 concepts, got {len(concepts)}")

    all_embeddings = []
    if args.backend == "transformers":
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(args.model)
        model = CLIPTextModel.from_pretrained(args.model).to(args.device).eval()
        with torch.no_grad():
            for concept in concepts:
                prompts = [template.format(concept) for template in CONCEPT_PROMPT_TEMPLATES]
                tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
                out = model(**tokens)
                emb = out.pooler_output
                all_embeddings.append(emb.detach().cpu())
    else:
        import open_clip

        model_name = "ViT-L-14" if args.model == "openai/clip-vit-large-patch14" else args.model
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(args.device).eval()
        with torch.no_grad():
            for concept in concepts:
                prompts = [template.format(concept) for template in CONCEPT_PROMPT_TEMPLATES]
                tokens = tokenizer(prompts).to(args.device)
                emb = model.encode_text(tokens)
                all_embeddings.append(emb.detach().cpu())

    text_embeddings = torch.stack(all_embeddings, dim=0)
    torch.save({"text_embeddings": text_embeddings, "concepts": concepts}, args.output)
    print(f"saved {tuple(text_embeddings.shape)} to {args.output}")


if __name__ == "__main__":
    main()
