import pyclapeyron as cl

# Check basic Julia works
cl.jl.println("Julia works!")

# Check pyclapeyron works
model = cl.BasicIdeal()
p = cl.pressure(model, 1., 300.)
print("Clapeyron works!")