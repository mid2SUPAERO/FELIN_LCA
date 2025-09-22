import brightway2 as bw
import openmdao.api as om
import numpy as np

from lca4mdao.component import LcaCalculationComponent
from lca4mdao.utilities import cleanup_parameters, setup_ecoinvent, setup_bw
from lca4mdao.variable import ExplicitComponentLCA

bw.projects.set_current("sellar_eco")
bw.bw2setup()
print(bw.projects.report())
print(bw.databases)

fp = r'C:\Users\joana\Desktop\Joana\FELIN\ecoinvent 3.8_cutoff_ecoSpold02\datasets'
if bw.Database("ecoinvent 3.8 cutoff").random() is None :
    ei = bw.SingleOutputEcospold2Importer(fp, "ecoinvent 3.8 cutoff")
    ei.apply_strategies()
    ei.statistics()
    ei.write_database()
else :
    print("ecoinvent 3.8 cutoff already imported")
ecoinvent = bw.Database("ecoinvent 3.8 cutoff")
print(ecoinvent.random())

# File path for ecoinvent datasets
#fp = 'C:\\Users\\joana\\Desktop\\Joana\\FELIN\\ecoinvent 3.8_cutoff_ecoSpold02\\datasets'
wood = ('ecoinvent 3.8 cutoff', 'a63dd664a99c9e82c192f8c50a9b4cfb')
steel = ('ecoinvent 3.8 cutoff', '580b7aea44c188e5958b4c6bd6ec515a')
method_key = ('ReCiPe Midpoint (H) V1.13', 'climate change', 'GWP100')

def build_data():
    sellar = bw.Database('sellar')
    sellar.register()
    sellar.delete(warn=False)
    sellar.new_activity('sellar_problem', name='sellar problem').save()


class SellarDis1(ExplicitComponentLCA):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=1.0)

        # Coupling output
        self.add_output('y1', lca_parent=("sellar", "sellar_problem"), lca_units='kilogram', lca_key=wood, val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1 ** 2 + z2 + x1 - 0.2 * y2


class SellarDis2(ExplicitComponentLCA):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', lca_parent=("sellar", "sellar_problem"), lca_units='kilogram', lca_key=steel, val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1 ** .5 + z1 + z2


class SellarLCA(LcaCalculationComponent):
    def setup(self):
        self.add_lca_output('GWP', {("sellar", "sellar_problem"): 1},
                            method_key=('ReCiPe Midpoint (H) V1.13', 'climate change', 'GWP100'))


class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'],
                            promotes_outputs=['y1'])
        cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'],
                            promotes_outputs=['y2'])

        cycle.set_input_defaults('x', 1.0)
        cycle.set_input_defaults('z', np.array([5.0, 2.0]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS(maxiter=100)

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
        self.add_subsystem('LCA', SellarLCA(), promotes=['*'])


if __name__ == '__main__':
    setup_bw("Example")
    setup_ecoinvent(fp, "ecoinvent 3.8 cutoff")
    build_data()
    cleanup_parameters()

    prob = om.Problem()
    prob.model = SellarMDA()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-8

    prob.model.add_design_var('x', lower=0, upper=10)
    prob.model.add_design_var('z', lower=0, upper=10)
    prob.model.add_objective('obj')
    prob.model.add_constraint('con1', upper=0)
    prob.model.add_constraint('con2', upper=0)

    # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    prob.model.approx_totals()

    prob.setup()
    prob.set_solver_print(level=0)

    prob.run_driver()

    print('minimum found at')
    print(prob.get_val('x')[0])
    print(prob.get_val('z'))

    print('Environmental parameters at minimum')
    print('wood: ' + str(prob.get_val('y1')[0]))
    print('steel: ' + str(prob.get_val('y2')[0]))

    print('minumum objective')
    print(prob.get_val('obj')[0])

    print('GWP at objective')
    print(prob.get_val('GWP')[0])

    prob = om.Problem()
    prob.model = SellarMDA()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-8

    prob.model.add_design_var('x', lower=0, upper=10)
    prob.model.add_design_var('z', lower=0, upper=10)
    prob.model.add_objective('GWP')
    prob.model.add_constraint('con1', upper=0)
    prob.model.add_constraint('con2', upper=0)

    # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    prob.model.approx_totals()

    prob.setup()
    prob.set_solver_print(level=0)

    prob.run_driver()

    print('minimum found at')
    print(prob.get_val('x')[0])
    print(prob.get_val('z'))

    print('Environmental parameters at minimum')
    print('wood: ' + str(prob.get_val('y1')[0]))
    print('steel: ' + str(prob.get_val('y2')[0]))

    print('minimum GWP')
    print(prob.get_val('GWP')[0])

    print('objective at minimum GWP')
    print(prob.get_val('obj')[0])
