import logging
from juliacall import Main as jl
import pyclapeyron as cl

logger = logging.getLogger(__name__)

def test_print_version():
    jl_version = cl.Clapeyron.VERSION
    cl_version = jl.pkgversion(cl.Clapeyron)
    
    logger.info(f"Julia version:        {jl_version}")
    logger.info(f"Clapeyron.jl version: {cl_version}")
