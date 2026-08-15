from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.hypergraph_builder import Hypergraph
from src.analysis.topology import compute_topology
from src.experiments.abm_pgg import Condition, run_simulation
from src.experiments.model_configs import get_api_key, get_api_url


class TopologyTests(unittest.TestCase):
    def test_egalitarian_hypergraph_has_maximal_his(self):
        graph = Hypergraph(
            nodes={"a", "b", "c"},
            hyperedges=[frozenset({"a", "b", "c"})],
            timestamps=[None],
        )
        report = compute_topology(graph, name="egalitarian", triadic_sample=10)
        self.assertAlmostEqual(report.his_mean, 1.0)
        self.assertAlmostEqual(report.frac_higher_order, 1.0)

    def test_hub_dominance_reduces_his(self):
        graph = Hypergraph(
            nodes={"hub", "a", "b", "c", "d", "e", "f"},
            hyperedges=[
                frozenset({"hub", "a", "b"}),
                frozenset({"hub", "c", "d"}),
                frozenset({"hub", "e", "f"}),
            ],
            timestamps=[None, None, None],
        )
        report = compute_topology(graph, name="hub", triadic_sample=10)
        self.assertLess(report.his_mean, 1.0)
        self.assertGreater(report.hyperdegree_gini, 0.0)


class SimulationTests(unittest.TestCase):
    def test_abm_is_reproducible_for_a_fixed_seed(self):
        kwargs = {
            "condition": Condition.C,
            "n_agents": 24,
            "n_rounds": 12,
            "seed_fraction": 0.125,
            "seed": 7,
            "avg_membership": 3,
        }
        first = run_simulation(**kwargs)
        second = run_simulation(**kwargs)
        np.testing.assert_array_equal(first.cooperation_rate, second.cooperation_rate)
        np.testing.assert_array_equal(first.norm_adoption_rate, second.norm_adoption_rate)
        self.assertTrue(np.all((first.cooperation_rate >= 0) & (first.cooperation_rate <= 1)))
        self.assertTrue(
            np.all((first.norm_adoption_rate >= 0) & (first.norm_adoption_rate <= 1))
        )


class CredentialConfigurationTests(unittest.TestCase):
    def test_gateway_configuration_has_no_embedded_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                get_api_key()
            with self.assertRaises(RuntimeError):
                get_api_url()


if __name__ == "__main__":
    unittest.main()
