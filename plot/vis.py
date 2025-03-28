import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch


# 学習後のアテンションの可視化 (変更)
def visualize_attention_on_image(device, model, image, original_image, class_names, save_path):
    model.eval()
    with torch.no_grad():
        # 入力画像をデバイスに移動し、バッチ次元を追加
        image_tensor = image.unsqueeze(0).to(device)
        
        output, attention_map = model(image_tensor, return_attention=True)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

        attention_map = attention_map.squeeze(0).cpu().numpy()
        
        combined_attn_map = np.sum(attention_map, axis=0)
        
        # ヒートマップのサイズを入力画像のサイズに合わせる
        original_height, original_width = original_image.shape[1], original_image.shape[2]
        attn_map_resized = cv2.resize(combined_attn_map, (original_width, original_height))

        # 0から1の範囲に正規化
        attn_map_normalized = (attn_map_resized - np.min(attn_map_resized)) / (np.max(attn_map_resized) - np.min(attn_map_resized) + 1e-8) # ゼロ除算回避
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_normalized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # オリジナル画像をRGBに変換
        original_image_rgb = (original_image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
        original_image_rgb = original_image_rgb.astype(np.uint8)

        # サイズが一致しているか確認 (デバッグ用)
        if original_image_rgb.shape[:2] != heatmap_rgb.shape[:2] or original_image_rgb.shape[2] != 3 or heatmap_rgb.shape[2] != 3:
            print(f"Error: Shape mismatch before overlaying.")
            print(f"Original Image Shape: {original_image_rgb.shape}")
            print(f"Heatmap Shape: {heatmap_rgb.shape}")
            return

        # ヒートマップをオーバーレイ
        alpha = 0.5  # 透明度
        overlay = cv2.addWeighted(original_image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

        plt.figure(figsize=(12, 6))

        # 左側：オリジナル画像
        plt.subplot(1, 2, 1)
        plt.imshow(original_image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        # 右側：Attention Overlay画像
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f'Top-Down Attention Overlay (Predicted: {predicted_class})')
        plt.axis("off")

        plt.savefig(save_path)
        plt.close()
