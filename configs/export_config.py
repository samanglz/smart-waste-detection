EXPORT_FORMAT = "onnx"

SIMPLIFY = True

DYNAMIC = False

# Note: DYNAMIC=True is useful for research but decreases performance.
# For production, keep DYNAMIC=False and resize images to fixed size (640).