# script.spec
block_cipher = None
 
a = Analysis(
    ['main.py'],
    pathex=["."],
    binaries=[],
    datas=[("vae_classifier_1024.onnx", "."), ("CAML_icon.ico", "."), ("CAML_icon.png", ".")],
    hiddenimports=["PIL", "PIL._tkinter_finder", "PIL._imagingtk", "PIL.Image", "pyarrow", "numexpr", "bottleneck", "classification_plot.py", "CustomSlider.py"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytorch", "ipython", "pytorch-cuda", "pytorch-mutex", "torchinfo", "torchtriton",
    "torchvision", "tensorboardx", "widgetsnbextension", "qt-main", "qt-webengine", "qtwebkit", 
    "jupyter_core", "jupyter_client", "ipykernel", "ipywidgets", "torch", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
 
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)
 
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CAML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon="CAML_icon.ico",
)
 
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CAML',
)