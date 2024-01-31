import unittest
import torch

from solvers.inner_solvers import minres, cg
import utils.utils as utils

# Run some rudimentary tests on the inner solvers.
torch.set_default_dtype(torch.float64)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEED = 0

class test_minres(unittest.TestCase):
    test_eps = 1e-8
    
    def test_small_diag_matrix(self):
        """
        Tests the system solver on a small diagonal PD system.
        """
        eta = 1e-4
        A = torch.diag(torch.tensor([1., 2., 3., 4.]))
        xstar = torch.ones((4,))
        b = A@xstar 
        Av = lambda v : A@v

        x, Dtype, _ = minres(Av, -b, eta)

        # At the moment just using defaults tolerance
        bhat = Av(x)
        r = bhat - b
        self.assertEqual(Dtype, "SOL")
        self.assertTrue(inexactness_test(A, b, x, eta))

        # in this case can test residual directly using lower bound on Hr and upper bound on Hx
        self.assertTrue(torch.linalg.norm(r) < eta*4*torch.linalg.norm(x))       

    def test_small_diag_npc(self):
        """
        Tests whether MINRES detects the NPC direction in a small diagonal matrix a manual test verifies that x^TAx < 0. 
        """
        eta = 0 # 0 tolerance should mean we dont terminate early
        A = torch.diag(torch.tensor([1., 2., -1., 3.]))
        Av = lambda v: A@v
        xstar = torch.ones((4,))

        b = A@xstar

        x, Dtype, _ = minres(Av, b, eta)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue((x.T@A@x).item() < 0)

    def test_large_PD_matrix(self):
        """
        Tests some random large PD matrices with known spectrum. Test passes if the inexact decrease condition ||Ar || <= eta ||A x|| is met by manually verifying. 
        """
        N = 100
        eta = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=1, maxeval=100)
        Av = lambda v: A@v

        x, Dtype, _ = minres(Av, -b, eta)

        self.assertEqual(Dtype, "SOL")
        self.assertTrue(inexactness_test(A, b, x, eta))

    def test_large_NPC_matrix(self):
        """
        Tests some random large matrix with one negative eigenvalue. Test passes if the Dtrpe is NPC and a manual test verifies that x^TAx < 0.
        """
        N = 100
        eta = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=-1, maxeval=100)
        Av = lambda v: A@v

        xstar = torch.randn((N,), generator=g)
        b = A@xstar 

        x, Dtype, _ = minres(Av, b, eta)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue((torch.dot(x, A@x)) < 0)

    def test_large_PSD_matrix(self): 
        """
        Tests large matrix which is almost rank deficient. Solution vector b is random normal and so b is not in Range(A).

        MINRES passes the test with the 0 eigenvalues if EPS=1e-6.
        """
        N = 100
        etas = [1e-2, 1e-4] # allows for mix of SOL and NPC

        g = torch.Generator()
        g.manual_seed(0)

        d = torch.linspace(1, 10, N-2)
        d = torch.concatenate([d, torch.tensor([-1e-6, -1e-6])])
        A = utils.build_symmetric_matrix_from_diag(d, g)
        Av = lambda v: A@v

        for j in range(100):
            eta = etas[j%2]
            b = torch.randn((N,), generator=g)
            b = b/torch.linalg.norm(b)

            x, Dtype, _ = minres(Av, -b, eta)        
            self.assertTrue(Dtype == "NPC" or "SOL")

            if Dtype =="NPC":
                self.assertTrue((x.T@A@x).item() < 1e-6)

            else:
                r = A@x-b
                self.assertTrue(inexactness_test(A, b, x, eta))


class test_cg(unittest.TestCase):
    test_eps = 1e-6

    def test_small_diag_matrix(self):
        """
        Tests the system solver on a small diagonal PD system.
        """
        acc = 1e-4
        A = torch.diag(torch.tensor([1., 2., 3., 4.]))
        xstar = torch.ones((4,))
        b = A@xstar 
        Av = lambda v: A@v

        x, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        # At the moment just using defaults tolerance
        bhat = Av(x)
        r = bhat - b
        self.assertEqual(Dtype, "SOL")
        self.assertTrue(residual_test(A, b, x, acc=0.1))
                
        # in this case can test residual directly using lower bound on Hr and upper bound on Hx

    def test_small_diag_npc(self):
        """
        Tests whether CG detects the NPC direction in a small diagonal matrix a manual test verifies that x^TAx < 0. 
        """
        eps = 1e-6
        acc = 1e-4 # small tolerance should mean we dont terminate early
        A = torch.diag(torch.tensor([1., 2., -1., 3.]))
        Av = lambda v: A@v
        xstar = torch.ones((4, ))

        b = A@xstar

        x, Dtype, _ = cg(Av, -b, eps, acc)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue((x.T@A@x).item() < 0)

    def test_large_PD_matrix(self):
        """
        Tests some random large PD matrices with known spectrum. Test passes if the inexact decrease condition ||Ar || <= eta ||A x|| is met by manually verifying. 
        """
        N = 100
        acc = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=1, maxeval=100)
        Av = lambda v: A@v

        x, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        self.assertEqual(Dtype, "SOL")
        self.assertTrue(residual_test(A, b, x, acc=0.1))

    def test_large_NPC_matrix(self):
        """
        Tests some random large matrix with one negative eigenvalue. Test passes if the Dtrpe is NPC and a manual test verifies that x^TAx < 0.
        """
        N = 100
        acc = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=-1, maxeval=100)
        Av = lambda v: A@v

        xstar = torch.randn((N,), generator=g)
        b = A@xstar 

        x, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue((x.T@A@x).item() < 0)

    def test_large_PSD_matrix(self): 
        """
        Tests a 100 random large matrices which is almost rank deficient. Solution vector b is random normal and so b is not in Range(A).

        MINRES passes the test with the 0 eigenvalues if EPS=1e-6.
        """
        N = 100
        accs = [1e-2, 1e-4] # allows for mix of SOL and NPC

        g = torch.Generator()
        g.manual_seed(0)

        # CG 
        d = torch.linspace(1, 10, N-2)
        d = torch.concatenate([d, torch.tensor([-1e-4, -1e-4])])
        A = utils.build_symmetric_matrix_from_diag(d, generator=g)
        Av = lambda v: A@v

        for j in range(100):
            acc = accs[j%2]
            b = torch.randn((N,), generator=g)

            b = b/torch.linalg.norm(b)

            x, Dtype, _ = cg(Av, -b, self.test_eps, acc)        
            self.assertTrue(Dtype == "NPC" or "SOL")

            if Dtype =="NPC":
                self.assertTrue(torch.dot(x, A@x) < 1e-6)

            else:
                self.assertTrue(residual_test(A, b, x, acc=0.1))
                
    
def inexactness_test(A, b, x, eta):
    r = A@x-b
    return torch.linalg.norm(A@r) <= eta*torch.linalg.norm(A@x)

def residual_test(A, b, x, acc=0.1):
    r = A@x-b
    return torch.linalg.norm(r) <= acc*torch.linalg.norm(b)


if __name__ == '__main__':
    unittest.main()