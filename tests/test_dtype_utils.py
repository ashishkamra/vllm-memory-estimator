from memory_estimator.dtype_utils import bytes_per_element
from memory_estimator.dtype_utils import normalise_dtype


def test_normalise_aliases():
    assert normalise_dtype("fp16").name == "float16"
    assert normalise_dtype("bf16").bytes == 2
    assert normalise_dtype("float8_e4m3fn").bytes == 1


def test_bytes_per_element_quantised():
    assert bytes_per_element("int4") == 0.5
    assert bytes_per_element("uint8") == 1
