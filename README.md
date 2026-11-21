<CodeBlock language="markdown" editable>
# WoundVision AI


AI-powered wound tissue segmentation research prototype.
Overview
WoundVision AI explores deep learning for segmentation
of wound tissue using PyTorch and U-Net ResNet34.
Tissue Classes
- Background
- Epithelial
- Slough
- Granulation
- Necrotic
- Other
Baseline Results
Validation: 30 images (patient-level split)
Metric	V0.1	V0.2
Macro Tissue Dice	24.18%	26.94%
Wound Dice	76.88%	78.69%

Technology
- Python
- PyTorch
- segmentation-models-pytorch
- Gradio
Disclaimer
Research prototype only. Not clinically validated.
Not intended for diagnosis or treatment decisions.
</CodeBlock>

Lalu jalankan:

<CodeBlock language="powershell" editable>
git add .gitignore README.md

git diff --cached --stat

git commit -m "chore: initialize WoundVision AI repository"

git status
</CodeBlock>

Kita sengaja belum memasukkan notebook dan app.py ke commit pertama. Keduanya akan kita audit dan rapikan terlebih dahulu.
Setelah ini, kita mulai refactoring
Berikut urutan yang gua sarankan:

<box gap={2}>
  <row align="center" gap={2}>
    <icon name="check-circle-2" color="success"/>
    <text>Repository dan struktur folder</text>
  </row>
  <row align="center" gap={2}>
    <icon name="check-circle-2" color="success"/>
    <text>Backup notebook dan checkpoint</text>
  </row>
  <row align="center" gap={2}>
    <icon name="circle-dot" color="info"/>
    <text>Commit pertama Git — sekarang</text>
  </row>
  <row align="center" gap={2}>
    <icon name="circle" color="secondary"/>
    <text>Refactoring `model.py` dan `data.py`</text>
  </row>
  <row align="center" gap={2}>
    <icon name="circle" color="secondary"/>
    <text>Evaluasi checkpoint V0.1 dan V0.2 di Windows</text>
  </row>
  <row align="center" gap={2}>
    <icon name="circle" color="secondary"/>
    <text>Push ke GitHub</text>
  </row>
</box>