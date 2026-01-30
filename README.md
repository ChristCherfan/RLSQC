# RLSQC

This repository implements the electronic Hamiltonian of simple molecules using the Voronoi finite-volume discretization scheme.

---

## An overview of the method is presented in the attached article

---

## File Descriptions

### `grid_generator.py`

Constructs a **3D real-space grid** using:

- Becke partitioning along the radial direction
- Lebedev quadrature for angular coordinates

The total number of grid points is controlled by:
- \( N_r \): number of radial points
- \( N_{ang} \): number of angular Lebedev points

---

### `voronoi.py`

Implements the **Voronoi finite-volume discretization** of differential operators
as described in **Appendix A**.

Includes matrix representations of:
- The Laplacian (with symmetrized transformation)
- First derivatives with respect to the nuclear–electron distance
- First derivatives with respect to the electron–electron distance

The first-derivative operators arise from the **transcorrelation transformation** (see appendix A of the article).

---

### `NTC_general.py`

Constructs the **non-transcorrelated Hamiltonian** by assembling the matrix terms. Diagonalizes the Hamiltonian, provides an estimate of the **ground-state energy** and plots the corresponding eigenstate in the radial direction.

---

### `general_molecule.py`

Same as `NTC_general.py` but for the transcorrelated Hamiltonian

---

### Note:
Larger grids were taken for more accurate results and the corresponding Hamiltonian were diagonalized using a Davidson solver (check part 5 of the article).

---

## License

MIT License


## File Structure

