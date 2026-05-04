import torch
from cascaide.data.dataset import CascadeDataset
from cascaide.encoding.base_image import BaseImageEncoder


def test_encode_decode_real():
    dataset = CascadeDataset(
        data_root="cascaide/dataset/defect_files_20260218/defect_files_20260218",
        max_samples=1
    )

    encoder = BaseImageEncoder(image_size=24)

    sample = dataset[0]

    vac = sample["vac_coords"]
    sia = sample["sia_coords"]

    print("Real sample shapes:")
    print("Vac:", vac.shape)
    print("SIA:", sia.shape)

    img = encoder.encode(vac, sia)

    print("Encoded shape:", img.shape)
    assert img.shape == (3, 24, 24)

    vac_rec, sia_rec = encoder.decode(img)

    print("Decoded shapes:")
    print("Vac:", vac_rec.shape)
    print("SIA:", sia_rec.shape)

    print("✅ Real-data encode-decode test passed!")


if __name__ == "__main__":
    test_encode_decode_real()