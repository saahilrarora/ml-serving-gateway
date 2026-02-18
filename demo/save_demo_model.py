"""Run this script once to generate the demo model file (demo/text_classifier.pt)."""

import sys
import os

# add project root to path so we can import from demo.models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from demo.models.text_classifier import TextClassifier

# where to save the serialized model
SAVE_PATH = os.path.join(os.path.dirname(__file__), "text_classifier.pt")


def main():
    # create the model with default params (input=10, hidden=32, classes=3)
    model = TextClassifier()
    model.eval()  # set to evaluation mode (disables dropout, batchnorm training behavior)

    # save the entire model (architecture + learned weights)
    torch.save(model, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

    # sanity check: load it back and run inference
    loaded = torch.load(SAVE_PATH, weights_only=False)
    loaded.eval()
    sample = TextClassifier.get_sample_input()   # random [1, 10] tensor
    output = loaded(sample)                       # forward pass
    print(f"Sanity check — input shape: {sample.shape}, output shape: {output.shape}")
    print(f"Sample output (probabilities): {output.detach().numpy()}")


if __name__ == "__main__":
    main()
