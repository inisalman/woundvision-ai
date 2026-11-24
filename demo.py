import gradio as gr

from inference import (
    load_model,
    predict,
    generate_overlay,
    calculate_tissue_percentage
)


# 1. Load model sekali saat aplikasi dimulai
model = load_model()


# 2. Fungsi analisis gambar
def analyze_wound(image, opacity):

    if image is None:
        raise gr.Error("Please upload a wound image.")

    # Prediksi segmentasi
    mask = predict(model, image)

    # Buat overlay
    overlay = generate_overlay(
        image,
        mask,
        alpha=opacity
    )

    # Hitung persentase jaringan
    analysis = calculate_tissue_percentage(mask)

    # Siapkan data tabel
    rows = []

    for tissue, result in analysis["tissues"].items():

        rows.append([
            tissue.capitalize(),
            result["pixels"],
            result["percentage"]
        ])

    return overlay, rows, analysis


# 3. Buat interface Gradio
with gr.Blocks(title="WoundVision AI") as demo:

    gr.Markdown(
        """
        # WoundVision AI
        ### AI-Powered Wound Tissue Segmentation

        U-Net ResNet34 | V5-Augmented

        Experimental research prototype.
        Not validated for clinical decision-making.
        """
    )

    with gr.Row():

        input_image = gr.Image(
            label="Original Wound Image",
            type="pil",
            sources=["upload"],
            height=400
        )

        output_image = gr.Image(
            label="Segmentation Result",
            type="pil",
            height=400,
            interactive=False
        )

    opacity = gr.Slider(
        minimum=0,
        maximum=1,
        value=0.5,
        step=0.05,
        label="Overlay Opacity"
    )

    with gr.Row():

        analyze_button = gr.Button(
            "Analyze Wound",
            variant="primary"
        )

        clear_button = gr.ClearButton(
            [input_image, output_image]
        )

    gr.Markdown("## Tissue Composition")

    tissue_table = gr.Dataframe(
        headers=[
            "Tissue Type",
            "Pixel Count",
            "Percentage (%)"
        ],
        datatype=["str", "number", "number"],
        interactive=False
    )

    with gr.Accordion("Raw Analysis JSON", open=False):

        json_output = gr.JSON(
            label="Analysis Result"
        )

    # 4. Hubungkan tombol dengan fungsi analisis
    analyze_button.click(
        fn=analyze_wound,
        inputs=[input_image, opacity],
        outputs=[
            output_image,
            tissue_table,
            json_output
        ]
    )

    clear_button.add([
        tissue_table,
        json_output
    ])


# 5. Jalankan aplikasi
if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )