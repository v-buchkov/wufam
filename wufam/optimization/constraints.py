############################################################################
### QPMwP - CONSTRAINTS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
import warnings

# Third party imports
import numpy as np
import pandas as pd


class Constraints:
    """params: list of identifiers - e.g., isocodes for the country ETFs"""

    def __init__(self, ids: list[str] = ["NA"]):
        self.ids = ids
        self.budget = {
            "Amat": None,
            "sense": None,
            "rhs": None,
        }  # i.e., sum of weights = 1, 'sense' -> =, < etc.,
        # 'rhs' any < const
        self.box = {"box_type": "NA", "lower": None, "upper": None}
        self.linear = {"G": None, "sense": None, "rhs": None}
        self.l1 = {}  # like L1-norm => absolute-value transform - needed for transaction costs constraints

    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, value):
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError("argument 'ids' has to be list[str].")
            self._ids = value
        else:
            raise ValueError("Input value must be a list.")
        return None

    def add_budget(self, rhs=1, sense="=") -> None:
        if self.budget.get("rhs") is not None:
            warnings.warn("Existing budget constraint is overwritten\n")

        a_values = pd.Series(np.ones(len(self.ids)), index=self.ids)
        self.budget = {"Amat": a_values, "sense": sense, "rhs": rhs}
        return None

    def add_box(self, box_type="LongOnly", lower=None, upper=None) -> None:
        def match_arg(x, lst):
            return [el for el in lst if x in el][0]

        box_type = match_arg(box_type, ["LongOnly", "LongShort", "Unbounded"])

        if box_type == "Unbounded":
            lower = float("-inf") if lower is None else lower
            upper = float("inf") if upper is None else upper
        elif box_type == "LongShort":
            lower = -1 if lower is None else lower
            upper = 1 if upper is None else upper
        elif box_type == "LongOnly":
            if lower is None:
                if upper is None:
                    lower = 0
                    upper = 1
                else:
                    lower = upper * 0
            else:
                if not np.isscalar(lower):
                    if any(lb < 0 for lb in lower):
                        raise ValueError(
                            "Inconsistent lower bounds for box_type 'LongOnly'. "
                            "Change box_type to LongShort or ensure that lower >= 0."
                        )

                upper = lower * 0 + 1 if upper is None else upper

        boxcon = {"box_type": box_type, "lower": lower, "upper": upper}

        if np.isscalar(boxcon["lower"]):
            boxcon["lower"] = pd.Series(
                np.repeat(float(boxcon["lower"]), len(self.ids)), index=self.ids
            )
        if np.isscalar(boxcon["upper"]):
            boxcon["upper"] = pd.Series(
                np.repeat(float(boxcon["upper"]), len(self.ids)), index=self.ids
            )

        if (boxcon["upper"] < boxcon["lower"]).any():
            raise ValueError(
                "Some lower bounds are higher than the corresponding upper bounds."
            )

        self.box = boxcon
        return None

    def add_linear(
        self,
        G: pd.DataFrame = None,
        g_values: pd.Series = None,
        sense: str = "=",
        rhs=None,
        name: str = None,
    ) -> None:
        if G is None:
            if g_values is None:
                raise ValueError("Either 'G' or 'g_values' must be provided.")
            else:
                G = pd.DataFrame(g_values).T.reindex(columns=self.ids).fillna(0)
                if name is not None:
                    G.index = [name]

        if isinstance(sense, str):
            sense = pd.Series([sense])

        if isinstance(rhs, (int, float)):
            rhs = pd.Series([rhs])

        if self.linear["G"] is not None:
            G = pd.concat([self.linear["G"], G], axis=0, ignore_index=False)
            sense = pd.concat([self.linear["sense"], sense], axis=0, ignore_index=False)
            rhs = pd.concat([self.linear["rhs"], rhs], axis=0, ignore_index=False)

        G.fillna(0, inplace=True)

        self.linear = {"G": G, "sense": sense, "rhs": rhs}
        return None

    def add_l1(self, name: str, rhs=None, x0=None, *args, **kwargs) -> None:
        if rhs is None:
            raise TypeError("argument 'rhs' is required.")
        con = {"rhs": rhs}
        if x0:
            con["x0"] = x0
        for i, arg in enumerate(args):
            con[f"arg{i}"] = arg
        for key, value in kwargs.items():
            con[key] = value
        self.l1[name] = con
        return None

    def to_GhAb(self, lbub_to_G: bool = False) -> dict[str, pd.DataFrame]:
        """Transform into the mathematical definition from the slides"""

        A = None
        b = None
        G = None
        h = None

        if self.budget["Amat"] is not None:
            if self.budget["sense"] == "=":
                A = np.array(self.budget["Amat"], dtype=float)
                b = np.array(self.budget["rhs"], dtype=float)
            else:
                G = np.array(self.budget["Amat"], dtype=float)
                h = np.array(self.budget["rhs"], dtype=float)

        if lbub_to_G:
            I = np.eye(len(self.selection))  # noqa: E741
            G_tmp = np.concatenate((-I, I), axis=0)
            h_tmp = np.concatenate((-self.box["lower"], self.box["upper"]), axis=0)
            G = np.vstack((G, G_tmp)) if (G is not None) else G_tmp
            h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        if self.linear["G"] is not None:
            Gmat = self.linear["G"].copy()
            rhs = self.linear["rhs"].copy()

            # Ensure that the system of inequalities is all '<='
            idx_geq = np.array(self.linear["sense"] == ">=")
            if idx_geq.sum() > 0:
                Gmat[idx_geq] = -Gmat[idx_geq]
                rhs[idx_geq] = -rhs[idx_geq]

            # Extract equality constraints
            idx_eq = np.array(self.linear["sense"] == "=")
            if idx_eq.sum() > 0:
                A_tmp = Gmat[idx_eq].to_numpy()
                b_tmp = rhs[idx_eq].to_numpy()
                A = np.vstack((A, A_tmp)) if A is not None else A_tmp
                b = np.concatenate((b, b_tmp), axis=None) if b is not None else b_tmp
                if idx_eq.sum() < Gmat.shape[0]:
                    G_tmp = Gmat[idx_eq == False].to_numpy()  # noqa: E712
                    h_tmp = rhs[idx_eq == False].to_numpy()  # noqa: E712
            else:
                G_tmp = Gmat.to_numpy()
                h_tmp = rhs.to_numpy()

            if "G_tmp" in locals():
                G = np.vstack((G, G_tmp)) if G is not None else G_tmp
                h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        # To ensure A and G are matrices (even with only 1 row)
        A = A.reshape(-1, A.shape[-1]) if A is not None else None
        G = G.reshape(-1, G.shape[-1]) if G is not None else None

        return {"G": G, "h": h, "A": A, "b": b}
