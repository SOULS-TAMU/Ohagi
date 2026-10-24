"""Dataset I/O and samplers: reproducibility, feasibility and malformed inputs."""

import numpy as np
import pytest
from ohagi.data.sampling import (
    load_csv_matrix,
    write_csv_matrix,
    split_train_val,
    sample_box,
    find_feasible_point,
    hit_and_run_samples,
)


@pytest.mark.parametrize("headers", [None, ["a", "b"]])
def test_csv_roundtrip(tmp_path, headers):
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    file = tmp_path / "nested" / "data.csv"
    write_csv_matrix(file, values, headers)
    np.testing.assert_array_equal(load_csv_matrix(file, ["a", "b"]), values)
    if headers:
        with file.open("a") as stream:
            stream.write("\n")
        np.testing.assert_array_equal(
            load_csv_matrix(file, ["b", "a"]), values[:, ::-1]
        )


@pytest.mark.parametrize(
    "text,columns",
    [
        ("", ["a"]),
        ("a\n", ["a"]),
        ("b\n1\n", ["a"]),
        ("a\nwrong\n", ["a"]),
        ("1,2\n3\n", ["a", "b"]),
        ("1\n", []),
    ],
)
def test_csv_rejects_invalid_data(tmp_path, text, columns):
    file = tmp_path / "data.csv"
    file.write_text(text)
    with pytest.raises(ValueError):
        load_csv_matrix(file, columns)


def test_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_matrix(tmp_path / "absent", ["a"])


@pytest.mark.parametrize("fraction", [0.001, 0.5, 0.999])
def test_split_is_disjoint_complete_and_reproducible(fraction):
    data = np.arange(20).reshape(10, 2)
    train, val = split_train_val(data, train_frac=fraction, seed=3)
    assert len(train) > 0 and len(val) > 0
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(10))
    other = split_train_val(data, train_frac=fraction, seed=3)
    np.testing.assert_array_equal(train, other[0])
    np.testing.assert_array_equal(val, other[1])


@pytest.mark.parametrize(
    "data,fraction",
    [
        (np.ones(4), 0.5),
        (np.ones((1, 2)), 0.5),
        (np.ones((2, 2)), 0),
        (np.ones((2, 2)), 1),
    ],
)
def test_invalid_split(data, fraction):
    with pytest.raises(ValueError):
        split_train_val(data, train_frac=fraction, seed=0)


def test_box_and_polytope_samples():
    low = np.array([-1.0, 2.0])
    high = np.array([3.0, 4.0])
    box = sample_box(low, high, num_samples=32, seed=7)
    assert np.all(box >= low) and np.all(box <= high)
    np.testing.assert_array_equal(box, sample_box(low, high, num_samples=32, seed=7))
    A = np.r_[np.eye(2), -np.eye(2), np.zeros((1, 2))]
    b = np.r_[high, -low, 1.0]
    start = find_feasible_point(A, b)
    assert np.all(A @ start <= b + 1e-10)
    for initial, burn, thin in [(None, None, None), (np.array([0.0, 3.0]), 2, 1)]:
        samples = hit_and_run_samples(
            A, b, num_samples=20, seed=8, start=initial, burn_in=burn, thinning=thin
        )
        assert samples.shape == (20, 2)
        assert np.all(samples @ A.T <= b + 1e-10)
        np.testing.assert_array_equal(
            samples,
            hit_and_run_samples(
                A, b, num_samples=20, seed=8, start=initial, burn_in=burn, thinning=thin
            ),
        )
    collapsed = hit_and_run_samples(
        [[1.0], [-1.0]], [0.0, 0.0], num_samples=3, seed=0, start=[0.0]
    )
    np.testing.assert_array_equal(collapsed, np.zeros((3, 1)))


@pytest.mark.parametrize(
    "A,b,start,n",
    [
        (np.ones(2), [1, 1], None, 2),
        (np.eye(2), [1], None, 2),
        (np.eye(2), [1, 1], None, 0),
        (np.eye(2), [1, 1], [2, 2], 2),
        (np.array([[-1.0]]), [0.0], [1.0], 2),
    ],
)
def test_invalid_polytope(A, b, start, n):
    with pytest.raises(ValueError):
        hit_and_run_samples(A, b, num_samples=n, seed=0, start=start)


def test_invalid_box_and_feasible_point():
    for low, high in [([0], [1, 2]), ([2], [1])]:
        with pytest.raises(ValueError):
            sample_box(low, high, num_samples=2, seed=0)
    for A, b in [
        (np.ones(2), [1, 1]),
        (np.eye(2), [1]),
        (np.array([[1.0], [-1.0]]), [0.0, -1.0]),
    ]:
        with pytest.raises(ValueError):
            find_feasible_point(A, b)
