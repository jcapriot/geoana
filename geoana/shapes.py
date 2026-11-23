"""
======================================================
Shapes (:mod:`geoana.shapes`)
======================================================
.. currentmodule:: geoana.shapes

The ``geoana.shapes`` module contains the basic geometric building blocks for
three dimensional objects.

Shape Classes
=============
.. autosummary::
  :toctree: generated/

  BasePrism
  BasePolyhedron
  BaseTetrahedron
"""
import numpy as np

from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from geoana.kernels.potential_field_polyhedron import (
    polyhedron_edge_log,
    triangle_solid_angle,
    face_edge_solid_angle,
    _canonical_face_normals,
    _edge_newtonian,
    _face_newtonian,
    _edge_dyads,
    _inside_fraction,
)

__all__ = [
    "BasePrism",
    "BasePolyhedron",
    "BaseTetrahedron",
]

# Faces whose unit normals differ by less than this are treated as coplanar.
_COPLANAR_TOL = 1e-12

# Number of (observation point, edge) pairs evaluated at once when summing a
# polyhedron's edge and face terms. It bounds the temporaries, so the memory
# used scales with the output, not with the number of points times the number
# of edges.
_BLOCK_SIZE = 2**18

# Generalized winding numbers this close to 0 or 1 decide whether a point is
# outside or inside a shell; anything else means the point is on its surface.
_WINDING_TOL = 1e-6


class BasePrism:
    """Class for basic geometry of a prism.

    The ``BasePrism`` class is used to define and validate basic geometry of an axis
    aligned prism in three dimensions.

    Parameters
    ----------
    min_location : (3,) numpy.ndarray of float
        Minimum location triple of the axis aligned prism
    max_location : (3,) numpy.ndarray of float
        Maximum location triple of the axis aligned prism
    """

    def __init__(self, min_location, max_location, **kwargs):
        super().__init__(**kwargs)
        self.min_location = min_location
        self.max_location = max_location
        if np.any(self.max_location <= self.min_location):
            raise ValueError("Max location must be strictly greater than the minimum location")

    @property
    def min_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._min_location

    @min_location.setter
    def min_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._min_location = vec

    @property
    def max_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._max_location

    @max_location.setter
    def max_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._max_location = vec

    @property
    def volume(self):
        """ The volume of the prism

        Returns
        -------
        float
        """
        return np.prod(self.max_location - self.min_location)

    @property
    def location(self):
        """ The center of the prism

        Returns
        -------
        (3,) numpy.ndarray of float
        """
        return 0.5 * (self.min_location + self.max_location)

    def _eval_def_int(self, func, x, y, z, cycle=0):
        "evaluate a definite integral (func) over the prism at x, y, z locations"

        x_min, y_min, z_min = self.min_location
        x_max, y_max, z_max = self.max_location

        x_min = x_min - x
        y_min = y_min - y
        z_min = z_min - z

        x_max = x_max - x
        y_max = y_max - y
        z_max = z_max - z

        for i in range(cycle):
            x_min, y_min, z_min = y_min, z_min, x_min
            x_max, y_max, z_max = y_max, z_max, x_max

        v000 = func(x_min, y_min, z_min)
        v001 = func(x_min, y_min, z_max)
        v010 = func(x_min, y_max, z_min)
        v011 = func(x_min, y_max, z_max)
        v100 = func(x_max, y_min, z_min)
        v101 = func(x_max, y_min, z_max)
        v110 = func(x_max, y_max, z_min)
        v111 = func(x_max, y_max, z_max)

        val = (v111 - v110 - v101 + v100 - v011 + v010 + v001 - v000)
        return val


class BasePolyhedron:
    r"""Class for basic geometry of a closed polyhedron with triangular faces.

    The ``BasePolyhedron`` class defines and validates a closed, consistently
    oriented triangulated surface, and evaluates the volume integrals of the
    Newtonian kernel :math:`1/|\mathbf{r} - \mathbf{r}'|` over the enclosed body
    in closed form, following Werner and Scheeres (1996):

    .. math::

        V(\mathbf{r}) &= \int_\Omega \frac{dV'}{|\mathbf{r} - \mathbf{r}'|}
            = \frac{1}{2}\sum_e \mathbf{r}_e \cdot \mathbf{E}_e \cdot \mathbf{r}_e L_e
            - \frac{1}{2}\sum_f \mathbf{r}_f \cdot \mathbf{F}_f \cdot \mathbf{r}_f \omega_f

        \nabla V &= -\sum_e \mathbf{E}_e \cdot \mathbf{r}_e L_e
            + \sum_f \mathbf{F}_f \cdot \mathbf{r}_f \omega_f

        \nabla\nabla V &= \sum_e \mathbf{E}_e L_e - \sum_f \mathbf{F}_f \omega_f

    where :math:`\mathbf{r}_e` and :math:`\mathbf{r}_f` point from the
    observation point to any point on edge :math:`e` or face :math:`f`,
    :math:`\mathbf{F}_f = \hat{n}_f \hat{n}_f^T` is the face dyad,
    :math:`\mathbf{E}_e = \hat{n}_A \hat{n}^A_e{}^T + \hat{n}_B \hat{n}^B_e{}^T` is
    the edge dyad built from the two faces sharing the edge and their in-plane
    edge normals, :math:`L_e = \ln\frac{r_1 + r_2 + \ell_e}{r_1 + r_2 - \ell_e}` is
    the edge's line integral, and :math:`\omega_f` is the signed solid angle the
    face subtends (van Oosterom and Strackee, 1983). The solid angles sum to
    :math:`4\pi` inside the body and to zero outside, so
    :math:`\nabla^2 V = -4\pi` inside and zero outside.

    Adjacent coplanar triangles are merged into planar polygonal faces, whose
    solid angles are summed edge by edge over their boundaries, so the
    diagonals of a triangulated flat face play no part.

    :math:`V` and :math:`\nabla V` are continuous everywhere. On the surface,
    :math:`\nabla\nabla V` takes the same values as
    :class:`geoana.gravity.Prism`: on a face, the mean of its limits from either
    side, and on an edge, where it diverges as :math:`-2\mathbf{E}_e\ln d` with
    the distance :math:`d` to the edge, the finite part left by dropping that
    term, :math:`L_e = \ln(4 r_1 r_2)`, or :math:`\ln(2\ell_e)` at the edge's
    end points. These conventions are additive, so bodies that tile a region sum
    to that region's values on their shared boundaries as well.

    The surface may consist of several closed shells, for example disjoint
    bodies or a body with cavities. The body is the region enclosed by an odd
    number of shells, so a shell inside one other shell is a cavity. Shells may
    touch at vertices and edges, but must not otherwise intersect.

    Parameters
    ----------
    vertices : (n_vertices, 3) array_like of float
        Vertex locations.
    faces : (n_faces, 3) array_like of int
        Vertex indices of each triangular face. The surface must be closed and
        consistently oriented within each shell: every edge is traversed equally
        often in each direction, by two faces at an ordinary edge, or by more
        where the surface touches itself along an edge. Each shell may be wound either
        inward or outward; they are reoriented on construction so that the
        normals point out of the body (into a cavity).

    References
    ----------
    Werner, R. A., and D. J. Scheeres (1996), Exterior gravitation of a
    polyhedron derived and compared with harmonic and mascon gravitation
    representations of asteroid 4769 Castalia, Celestial Mechanics and Dynamical
    Astronomy, 65, 313-344.

    van Oosterom, A., and J. Strackee (1983), The solid angle of a plane
    triangle, IEEE Transactions on Biomedical Engineering, BME-30(2), 125-126.
    """

    def __init__(self, vertices, faces, **kwargs):
        super().__init__(**kwargs)
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n_vertices, 3), got {vertices.shape}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (n_faces, 3), got {faces.shape}")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError(f"faces must be integer indices, got {faces.dtype}")
        if len(faces) < 4:
            raise ValueError(
                f"a closed polyhedron needs at least 4 faces, got {len(faces)}"
            )
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("faces reference a vertex index out of range")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("a face repeats a vertex index")
        self._vertices = vertices
        self._faces = faces.astype(np.intp)
        self._build_edges()
        self._orient_shells()
        if self.volume <= 0:
            raise ValueError("polyhedron has zero volume")
        self._build_dyads()

    def _directed_edges(self):
        """Each face's three directed edges, face by face in blocks: (3 n_faces, 2)."""
        f = self._faces
        return np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])

    def _build_edges(self):
        """Validate the surface, find its shells, and pair the faces at each edge.

        Every undirected edge must be traversed equally often in each direction.
        Faces are first joined into shells across ordinary (two-face) edges.
        Then, within each shell, each face traversing an edge forwards is paired
        with one traversing it backwards. Where the surface touches itself along
        an edge, that edge appears once per pair. The edge dyad is a sum of
        per-face terms, so the pairing does not change the result.
        """
        faces = self._faces
        n_faces = len(faces)
        n_v = len(self._vertices)
        directed = self._directed_edges()
        owner = np.tile(np.arange(n_faces), 3)
        lo = np.minimum(directed[:, 0], directed[:, 1])
        hi = np.maximum(directed[:, 0], directed[:, 1])
        forward = directed[:, 0] < directed[:, 1]
        key = lo * n_v + hi

        _, edge_id, counts = np.unique(key, return_inverse=True, return_counts=True)
        n_forward = np.bincount(edge_id, weights=forward, minlength=len(counts))
        unbalanced = 2 * n_forward != counts
        if np.any(unbalanced):
            e = np.flatnonzero(unbalanced[edge_id])[0]
            a, b = directed[e]
            if counts[edge_id[e]] == 1:
                raise ValueError(
                    f"edge {a}-{b} belongs to only one face; surface is not closed"
                )
            if counts[edge_id[e]] == 2:
                raise ValueError(
                    f"edge {a}->{b} is traversed twice in the same direction; "
                    "faces are not consistently oriented"
                )
            raise ValueError(
                f"edge {a}-{b} is traversed {int(n_forward[edge_id[e]])} times one way "
                f"and {int(counts[edge_id[e]] - n_forward[edge_id[e]])} times the other; "
                "surface is not closed or not consistently oriented"
            )

        # Shells: faces connected across ordinary (two-face) edges.
        ordinary = counts[edge_id] == 2
        order = np.argsort(edge_id[ordinary], kind="stable")
        pair_owner = owner[ordinary][order].reshape(-1, 2)
        graph = coo_array(
            (np.ones(len(pair_owner)), (pair_owner[:, 0], pair_owner[:, 1])),
            shape=(n_faces, n_faces),
        )
        n_shells, shell = connected_components(graph, directed=False)

        # Pair forward with backward traversals of each edge within each shell.
        # Sorting by (edge, shell, direction) puts each group's k backward
        # traversals before its k forward ones.
        sort = np.lexsort((forward, shell[owner], edge_id))
        group_key = edge_id[sort] * n_shells + shell[owner][sort]
        starts = np.flatnonzero(np.r_[True, group_key[1:] != group_key[:-1]])
        sizes = np.diff(np.r_[starts, len(sort)])
        n_fwd_group = np.add.reduceat(forward[sort].astype(np.intp), starts)
        if np.any(2 * n_fwd_group != sizes):
            g = np.flatnonzero(2 * n_fwd_group != sizes)[0]
            a, b = directed[sort[starts[g]]]
            raise ValueError(
                f"the faces meeting at edge {a}-{b} do not form closed shells; "
                "the surface is not a valid boundary of a solid"
            )
        half = np.repeat(sizes // 2, sizes // 2)
        first = np.repeat(starts, sizes // 2)
        rank = np.arange(len(first)) - np.repeat(
            np.cumsum(sizes // 2) - sizes // 2, sizes // 2
        )
        backward_idx = sort[first + rank]
        forward_idx = sort[first + half + rank]

        self._shell = shell
        self._edges = directed[forward_idx]
        # Indices into the directed edges: face A traverses the edge a -> b,
        # face B traverses it b -> a.
        self._edge_directed = np.stack([forward_idx, backward_idx], axis=-1)
        self._edge_faces = owner[self._edge_directed]

    def _orient_shells(self):
        """Wind each shell so its normals point out of the body.

        A shell enclosed by an even number of other shells bounds material and
        is wound outward; one enclosed by an odd number bounds a cavity and is
        wound inward, i.e. pointing out of the material into the cavity.
        """
        shell = self._shell
        n_shells = shell.max() + 1
        v = self._vertices[self._faces]
        signed = np.bincount(
            shell,
            weights=np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])),
            minlength=n_shells,
        )
        if np.any(signed == 0):
            raise ValueError("polyhedron has a shell with zero volume")

        depth = np.zeros(n_shells, dtype=np.intp)
        if n_shells > 1:
            lo = np.full((n_shells, 3), np.inf)
            hi = np.full((n_shells, 3), -np.inf)
            np.minimum.at(lo, shell, v.min(axis=1))
            np.maximum.at(hi, shell, v.max(axis=1))
            # Shell s can only be inside shell t if t's bounding box holds s's.
            within = np.all(lo[:, None] >= lo[None], axis=-1) & np.all(
                hi[:, None] <= hi[None], axis=-1
            )
            np.fill_diagonal(within, False)
            for s, t in zip(*np.nonzero(within)):
                depth[s] += self._shell_contains(t, s)

        want_positive = depth % 2 == 0
        flip = (signed > 0) != want_positive
        if np.any(flip):
            flip_faces = flip[shell]
            self._faces[flip_faces] = self._faces[flip_faces, ::-1]
            self._build_edges()

    def _shell_contains(self, t, s):
        """Whether shell ``s`` lies inside shell ``t``.

        Shells do not cross, so it is enough to test points of ``s`` with the
        generalized winding number of ``t``, skipping points on ``t``'s surface
        (where the shells touch) until one is clearly inside or outside.
        """
        v = self._vertices
        tri = v[self._faces[self._shell == t]]
        points = v[self._faces[self._shell == s]].mean(axis=1)
        for block in np.array_split(points, max(1, len(points) // 16)):
            r = tri[None, :, :, :] - block[:, None, None, :]
            winding = np.abs(
                triangle_solid_angle(*np.moveaxis(r.reshape(*r.shape[:2], 9), -1, 0)).sum(axis=-1)
            ) / (4 * np.pi)
            decided = (winding < _WINDING_TOL) | (np.abs(winding - 1) < _WINDING_TOL)
            if np.any(decided):
                return bool(winding[np.argmax(decided)] > 0.5)
        raise ValueError(
            f"could not tell whether shell {s} is inside shell {t}; "
            "shells must not overlap"
        )

    def _build_dyads(self):
        v = self._vertices
        f = self._faces
        n, order, area2 = _canonical_face_normals(v[f])
        if np.any(area2 == 0):
            raise ValueError("polyhedron has a degenerate (zero-area) face")
        face_ref = np.take_along_axis(f, order, axis=-1)[:, 0]

        n_a = n[self._edge_faces[:, 0]]
        n_b = n[self._edge_faces[:, 1]]
        # An edge between coplanar faces (e.g. the diagonal of a triangulated
        # flat face) has a vanishing dyad, so it contributes nothing. Its dyad
        # is only roundoff, which would make 0 * inf on the edge itself.
        coplanar = np.linalg.norm(np.cross(n_a, n_b), axis=-1) < _COPLANAR_TOL
        coplanar &= np.einsum("ij,ij->i", n_a, n_b) > 0

        # Merge coplanar faces into planar polygons.
        n_faces = len(f)
        ef = self._edge_faces[coplanar]
        graph = coo_array(
            (np.ones(len(ef)), (ef[:, 0], ef[:, 1])), shape=(n_faces, n_faces)
        )
        n_groups, group = connected_components(graph, directed=False)
        counts = np.bincount(group, minlength=n_groups)
        area_n = np.zeros((n_groups, 3))
        np.add.at(area_n, group, n * area2[:, None])
        n_g = area_n / np.linalg.norm(area_n, axis=-1)[:, None]
        single = counts[group] == 1
        n_g[group[single]] = n[single]
        # Reference vertex of each polygon: its lexicographically first vertex.
        rank = np.empty(len(v), dtype=np.intp)
        rank[np.lexsort((v[:, 2], v[:, 1], v[:, 0]))] = np.arange(len(v))
        ref_rank = np.full(n_groups, len(v))
        np.minimum.at(ref_rank, group, rank[face_ref])
        self._face_ref = np.argsort(rank)[ref_rank]
        self._face_normals = n_g
        self._face_dyads = n_g[:, :, None] * n_g[:, None, :]

        # Directed boundary edges of the polygons, grouped by polygon: all but
        # the two traversals of each edge merged away between coplanar faces.
        directed = self._directed_edges()
        owner = group[np.tile(np.arange(n_faces), 3)]
        boundary = np.ones(len(directed), dtype=bool)
        boundary[self._edge_directed[coplanar].ravel()] = False
        directed, owner = directed[boundary], owner[boundary]
        sort = np.argsort(owner, kind="stable")
        self._face_edges = directed[sort]
        self._face_edge_group = owner[sort]
        self._face_edge_starts = np.searchsorted(owner[sort], np.arange(n_groups))

        self._edges = self._edges[~coplanar]
        self._edge_faces = self._edge_faces[~coplanar]
        self._edge_directed = self._edge_directed[~coplanar]
        n_a, n_b = n_a[~coplanar], n_b[~coplanar]

        a, b = self._edges[:, 0], self._edges[:, 1]
        # Face A traverses the edge a->b, face B b->a.
        self._edge_dyads, self._edge_lengths = _edge_dyads(v[b] - v[a], n_a, n_b)

    @property
    def vertices(self):
        """Vertex locations.

        Returns
        -------
        (n_vertices, 3) numpy.ndarray of float
        """
        return self._vertices

    @property
    def faces(self):
        """Vertex indices of each triangular face, ordered so normals point outward.

        Returns
        -------
        (n_faces, 3) numpy.ndarray of int
        """
        return self._faces

    @property
    def volume(self):
        """The volume of the polyhedron.

        Returns
        -------
        float
        """
        v = self._vertices[self._faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    @property
    def location(self):
        """The centroid of the polyhedron.

        Returns
        -------
        (3,) numpy.ndarray of float
        """
        v = self._vertices[self._faces]
        w = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
        return (w[:, None] * v.sum(axis=1)).sum(axis=0) / (4.0 * w.sum())

    def _evaluate(self, xyz, order=None, inside_fraction=False):
        """Newtonian integrals and/or the inside fraction at (..., 3) points.

        ``order`` is None (skip), 0 (V), 1 (grad V) or 2 (Hessian of V). The
        edge and face terms are summed in blocks of about ``_BLOCK_SIZE``
        (point, edge) pairs into the (n_points, ...) outputs, so the memory used
        scales with the output, independent of the body's number of edges.

        Returns ``(integrals, fraction)``, either of which is None when not
        requested. The fraction of a small sphere about each point inside the
        body is one inside, zero outside, and on the surface the solid angle of
        the body seen from the point over 4 pi, e.g. 1/2 on a face.
        """
        if order not in (None, 0, 1, 2):
            raise ValueError(f"order must be 0, 1 or 2, got {order}")
        shape = xyz.shape[:-1]
        pts = xyz.reshape(-1, 3)
        n_pts = len(pts)
        block = max(1, _BLOCK_SIZE // max(n_pts, 1))
        v = self._vertices

        out = None
        if order is not None:
            out = np.zeros((n_pts,) + (3,) * order)
            for s in range(0, len(self._edges), block):
                e = slice(s, s + block)
                r_a = v[self._edges[e, 0]] - pts[:, None, :]
                r_b = v[self._edges[e, 1]] - pts[:, None, :]
                L = polyhedron_edge_log(*np.moveaxis(r_a, -1, 0), *np.moveaxis(r_b, -1, 0))
                out += _edge_newtonian(
                    order, r_a, r_b, L, self._edge_lengths[e], self._edge_dyads[e]
                )

        omega_sum = np.zeros(n_pts) if inside_fraction else None
        # Blocks of whole polygons, holding about ``block`` boundary edges each.
        n_poly = len(self._face_normals)
        starts = np.r_[self._face_edge_starts, len(self._face_edges)]
        p0 = 0
        while p0 < n_poly:
            p1 = np.searchsorted(starts, starts[p0] + block, side="right") - 1
            p1 = min(max(p1, p0 + 1), n_poly)
            e0, e1 = starts[p0], starts[p1]
            n = self._face_normals[p0:p1]
            h = np.einsum("npi,pi->np", v[self._face_ref[p0:p1]] - pts[:, None, :], n)
            local = self._face_edge_group[e0:e1] - p0
            r1 = v[self._face_edges[e0:e1, 0]] - pts[:, None, :]
            r2 = v[self._face_edges[e0:e1, 1]] - pts[:, None, :]
            omega = face_edge_solid_angle(
                *np.moveaxis(r1, -1, 0),
                *np.moveaxis(r2, -1, 0),
                *n[local].T,
                h[:, local],
            )
            omega = np.add.reduceat(omega, starts[p0:p1] - e0, axis=-1)
            if out is not None:
                out += _face_newtonian(order, h, omega, n, self._face_dyads[p0:p1])
            if inside_fraction:
                omega_sum += omega.sum(axis=-1)
            p0 = p1

        if out is not None:
            out = out.reshape(*shape, *out.shape[1:])
        frac = None
        if inside_fraction:
            frac = _inside_fraction(omega_sum[:, None]).reshape(shape)
        return out, frac

    def _newtonian_integrals(self, xyz, order):
        """V, grad V or the Hessian of V at observation points.

        ``order`` is 0 (V), 1 (grad V) or 2 (Hessian). ``xyz`` is (..., 3).
        """
        return self._evaluate(xyz, order)[0]

    def _inside_fraction(self, xyz):
        """Fraction of a small sphere about each (..., 3) point inside the body.

        One inside, zero outside, and on the surface the solid angle of the body
        seen from the point over 4 pi, e.g. 1/2 on a face.
        """
        return self._evaluate(xyz, inside_fraction=True)[1]


class BaseTetrahedron(BasePolyhedron):
    """Class for basic geometry of a tetrahedron.

    Parameters
    ----------
    vertices : (4, 3) array_like of float
        The four vertex locations, in any order.
    """

    def __init__(self, vertices, **kwargs):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (4, 3):
            raise ValueError(f"vertices must have shape (4, 3), got {vertices.shape}")
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
        super().__init__(vertices, faces, **kwargs)
