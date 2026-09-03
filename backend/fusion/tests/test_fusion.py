import numpy as np
import matplotlib.pyplot as plt
from fusion.sar_optical_fusion import execute_optical_sar_fusion

def run_test():
    # 1. Generate dummy Optical RGB (512x512x3) and SAR (512x512) arrays
    mock_optical = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mock_sar = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    # 2. Execute fusion pipeline
    result = execute_optical_sar_fusion(mock_optical, mock_sar, "Identify built-up regions.")

    # 3. Print results to terminal
    print("--- FUSION TEST RESULTS ---")
    print("Status:", result["status"])
    print("Composite Shape:", result["composite_image"].shape)
    print("\nGenerated Prompt:\n", result["generated_prompt"])

    # 4. Display composite image preview window
    plt.imshow(result["composite_image"])
    plt.title("Optical-SAR False-Color Composite Preview")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_test()