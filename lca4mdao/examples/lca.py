import bw2data as bd
import brightway2 as bw
import bw2io as bi
from bw2calc import LCA
import pandas as pd
from peewee import IntegrityError
import openmdao.api as om
import numpy as np

from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp, ScipyOptimizeDriver
from lca4mdao.component import LcaCalculationComponent
from lca4mdao.utilities import cleanup_parameters, setup_ecoinvent, setup_bw
from lca4mdao.variable import ExplicitComponentLCA

bw.projects.set_current("lca")
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

carbon_fibre = ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b')
aluminium_almg3 = ('ecoinvent 3.8 cutoff', '3d66c7f5f8d813a5b63b2d19a41ec763')
aluminium_alli = ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7')
titanium = ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072')
nickel = ('ecoinvent 3.8 cutoff', '6f592c599b70d14247116fdf44a0824a')
steel = ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2')
electronic_active = ('ecoinvent 3.8 cutoff', '52c4f6d2e1ec507b1ccc96056a761c0d')
electronic_passive = ('ecoinvent 3.8 cutoff', 'b1b65fe4d00b29f2299c72b894a3c0a0')
wire = ('ecoinvent 3.8 cutoff', 'f8586b86fe8ac595be9f6b18e9b94488')
battery = ('ecoinvent 3.8 cutoff', 'b2feecd5152754c08303bc84dc371b68')
motor = ('ecoinvent 3.8 cutoff', '0a45c922ec9f5a8345c88fb3ecc28b6f')
oxygen = ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456')
hydrogen = ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39')
transport = ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083')
electricity = ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643')


def build_data():
    felin = bw.Database('felin')
    felin.register()
    felin.delete(warn=False)
    felin.new_activity('felin_manufacture', name='felin_manufacture').save()
    felin.new_activity('felin_launch', name='felin_launch').save()


