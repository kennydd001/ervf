from moe_lab.shared_basis import shared_basis_parameter_count


def test_shared_basis_parameter_count() -> None:
    assert shared_basis_parameter_count(2048, 64, 8) == 1_064_960
