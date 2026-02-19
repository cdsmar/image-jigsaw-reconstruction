import os
import cv2
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tkinter.simpledialog as simpledialog
from pathlib import Path

# Automatically get the current user's home directory
HOME_DIR = str(Path.home())

BASE_DIR = os.path.join(HOME_DIR, "Downloads", "Project-img")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
REFERENCE_DIR = os.path.join(BASE_DIR, "Correct Images", "correct")
FINAL_DIR = os.path.join(BASE_DIR, "final_result")
VISUALS_DIR = os.path.join(BASE_DIR, "Visuals", "FullPipeline")

STANDARD_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-trained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((STANDARD_SIZE, STANDARD_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_deep_feature(img):
    with torch.no_grad():
        x = transform(img).unsqueeze(0).to(device)
        feat = resnet(x)
        feat = feat.squeeze().cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)  # normalize
    return feat

class PuzzleSlicer:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def slice_and_save(self, image_path, grid_size, category):
        img = cv2.imread(image_path)
        if img is None:
            return
        img = cv2.resize(img, (STANDARD_SIZE, STANDARD_SIZE))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(self.output_root, category, base_name)
        os.makedirs(save_dir, exist_ok=True)

        step = STANDARD_SIZE // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                piece = img[i*step:(i+1)*step, j*step:(j+1)*step]
                cv2.imwrite(os.path.join(save_dir, f"piece_{i}_{j}.jpg"), piece)

    def run(self):
        categories = {"2x2":2, "4x4":4, "8x8":8}
        for cat, grid in categories.items():
            src = os.path.join(self.input_root, cat)
            if not os.path.exists(src):
                continue
            images = glob.glob(os.path.join(src, "*.jpg"))
            for img_path in images:
                self.slice_and_save(img_path, grid, cat)

class PuzzleSolver:
    def __init__(self, input_root, ref_root, output_root):
        self.input_root = input_root
        self.ref_root = ref_root
        self.output_root = output_root
        self.log = {
            "timestamp": datetime.now().isoformat(),
            "total_solved": 0,
            "total_failed": 0,
            "details": []
        }
        self.shuffled_images = {}

    def extract_reference_patches(self, ref_img, grid):
        ref_img = cv2.resize(ref_img, (STANDARD_SIZE, STANDARD_SIZE))
        step = STANDARD_SIZE // grid
        patches = []
        for i in range(grid):
            for j in range(grid):
                patches.append(ref_img[i*step:(i+1)*step, j*step:(j+1)*step])
        return patches

    def solve_single(self, pieces, ref_patches, grid, folder_name, category):
        num = grid * grid
        step = STANDARD_SIZE // grid
        pieces = [cv2.resize(p, (step, step)) for p in pieces]

        # Shuffle pieces for visualization
        shuffled_indices = list(range(len(pieces)))
        random.shuffle(shuffled_indices)
        shuffled_pieces = [pieces[i] for i in shuffled_indices]

        # Store shuffled image
        unique_key = f"{category}_{folder_name}"
        self.shuffled_images[unique_key] = self.assemble(shuffled_pieces, list(range(num)), grid)

        # Precompute deep features for pieces and reference patches
        piece_feats = [extract_deep_feature(p) for p in pieces]
        ref_feats = [extract_deep_feature(r) for r in ref_patches]

        # Compute LAB color and deep feature similarity
        cost = np.zeros((num, num))
        for i in range(num):
            p_lab = cv2.cvtColor(pieces[i], cv2.COLOR_BGR2LAB).astype(float)
            for j in range(num):
                r_lab = cv2.cvtColor(ref_patches[j], cv2.COLOR_BGR2LAB).astype(float)
                lab_sim = -np.mean(np.abs(p_lab - r_lab))
                deep_sim = np.dot(piece_feats[i], ref_feats[j])
                # Combine similarities (weights can be tuned)
                cost[i][j] = 0.5 * lab_sim + 0.5 * deep_sim

        # Hungarian algorithm for assignment
        rows, cols = linear_sum_assignment(cost, maximize=True)
        arrangement = [0]*num
        for r, c in zip(rows, cols):
            arrangement[c] = r
        return arrangement

    def assemble(self, pieces, arrangement, grid):
        step = STANDARD_SIZE // grid
        canvas = np.zeros((STANDARD_SIZE, STANDARD_SIZE, 3), dtype=np.uint8)
        for idx, p_idx in enumerate(arrangement):
            r = idx // grid
            c = idx % grid
            piece = cv2.resize(pieces[p_idx], (step, step))
            canvas[r*step:(r+1)*step, c*step:(c+1)*step] = piece
        return canvas

    def run(self):
        categories = ["2x2","4x4","8x8"]
        for cat in categories:
            input_cat = os.path.join(self.input_root, cat)
            output_cat = os.path.join(self.output_root, cat)
            os.makedirs(output_cat, exist_ok=True)

            if not os.path.exists(input_cat):
                continue

            grid = int(cat[0])
            folders = os.listdir(input_cat)

            for folder in folders:
                puzzle_dir = os.path.join(input_cat, folder)
                ref_path = os.path.join(self.ref_root, f"{folder}.jpg")
                if not os.path.exists(ref_path):
                    self.log["total_failed"] += 1
                    continue

                ref_img = cv2.imread(ref_path)
                if ref_img is None:
                    self.log["total_failed"] += 1
                    continue

                piece_files = sorted(os.listdir(puzzle_dir))
                pieces = []
                for f in piece_files:
                    img = cv2.imread(os.path.join(puzzle_dir, f))
                    if img is not None:
                        pieces.append(img)

                if len(pieces) != grid*grid:
                    self.log["total_failed"] += 1
                    continue

                ref_patches = self.extract_reference_patches(ref_img, grid)
                arrangement = self.solve_single(pieces, ref_patches, grid, folder, cat)
                final_img = self.assemble(pieces, arrangement, grid)

                # Add grid lines
                step = STANDARD_SIZE // grid
                for i in range(1, grid):
                    final_img[i*step-1:i*step+1, :] = 255
                    final_img[:, i*step-1:i*step+1] = 255

                cv2.imwrite(os.path.join(output_cat, f"solved_{folder}.jpg"), final_img)
                self.log["total_solved"] += 1

        with open(os.path.join(self.output_root, "solving_log.json"), "w") as f:
            json.dump(self.log, f, indent=4)

def visualize_pipeline(solver):
    categories = {"2x2":2,"4x4":4,"8x8":8}
    fig, axes = plt.subplots(3,3, figsize=(18,15))
    fig.suptitle("FULL PUZZLE PIPELINE", fontsize=22)
    row = 0

    for cat, grid in categories.items():
        solved_cat = os.path.join(FINAL_DIR, cat)
        if not os.path.exists(solved_cat):
            continue

        solved_files = [f for f in os.listdir(solved_cat) if f.startswith("solved_")]
        if not solved_files:
            continue

        rand = random.choice(solved_files)
        img_id = rand.replace("solved_","").replace(".jpg","")

        orig = cv2.imread(os.path.join(DATA_DIR, cat, f"{img_id}.jpg"))
        solved = cv2.imread(os.path.join(solved_cat, rand))
        unique_key = f"{cat}_{img_id}"
        shuffled = solver.shuffled_images.get(unique_key)

        if orig is None or solved is None or shuffled is None:
            continue

        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        solved = cv2.cvtColor(solved, cv2.COLOR_BGR2RGB)
        shuffled = cv2.cvtColor(shuffled, cv2.COLOR_BGR2RGB)

        axes[row,0].imshow(orig)
        axes[row,0].set_title(f"Original ({cat})")
        axes[row,0].axis("off")

        axes[row,1].imshow(shuffled)
        axes[row,1].set_title("Shuffled")
        axes[row,1].axis("off")

        axes[row,2].imshow(solved)
        axes[row,2].set_title("Solved with grid")
        axes[row,2].axis("off")

        row += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALS_DIR, f"pipeline_{timestamp}.png")
    plt.savefig(save_path)
    plt.show()

def interactive_test():
    # Initialize Tkinter
    root = tk.Tk()
    root.title("Puzzle Solver Interactive Test")
    root.geometry("400x100")


    # Ask user to select an image
    img_path = filedialog.askopenfilename(title="Select an image to test",
                                          filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
    if not img_path:
        print("No image selected.")
        return
    root.withdraw()
    # Ask user grid size
    grid = simpledialog.askinteger("Grid Size", "Enter grid size (2,4,8):", minvalue=2, maxvalue=8)
    if grid is None:
        print("No grid size entered.")
        return

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image.")
        return

    img = cv2.resize(img, (STANDARD_SIZE, STANDARD_SIZE))
    step = STANDARD_SIZE // grid

    # Slice pieces
    pieces = [img[i * step:(i + 1) * step, j * step:(j + 1) * step] for i in range(grid) for j in range(grid)]

    # Shuffle pieces
    shuffled_indices = list(range(len(pieces)))
    random.shuffle(shuffled_indices)
    shuffled_pieces = [pieces[i] for i in shuffled_indices]

    # Solve using deep features
    solver = PuzzleSolver("", "", "")
    ref_patches = pieces.copy()
    arrangement = solver.solve_single(shuffled_pieces, ref_patches, grid, "interactive", "interactive")
    solved_img = solver.assemble(shuffled_pieces, arrangement, grid)

    # Add grid lines
    for i in range(1, grid):
        solved_img[i * step - 1:i * step + 1, :] = 255
        solved_img[:, i * step - 1:i * step + 1] = 255

    # Show images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shuffled_rgb = cv2.cvtColor(solver.assemble(shuffled_pieces, list(range(len(pieces))), grid), cv2.COLOR_BGR2RGB)
    solved_rgb = cv2.cvtColor(solved_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(shuffled_rgb)
    plt.title("Shuffled")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(solved_rgb)
    plt.title("Solved")
    plt.axis("off")
    plt.show()

# Run interactive test
if __name__ == "__main__":
    interactive_test()