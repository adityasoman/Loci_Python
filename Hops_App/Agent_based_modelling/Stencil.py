# Import Statements
import topogenesis as tg
import numpy as np

# creating neighborhood definition
stencil_von_neumann = tg.create_stencil("von_neumann", 1, 1)
stencil_von_neumann.set_index([0,0,0], 0)

# creating neighborhood definition
stencil_moore = tg.create_stencil("moore", 1, 1)
stencil_moore.set_index([0,0,0], 0)

# creating neighborhood definition 
stencil_squareness_moore = tg.create_stencil("moore", 1, 1)
# Reshaping the moore neighbourhood
stencil_squareness_moore[0,:,:] = 0 
stencil_squareness_moore[2,:,:] = 0
stencil_squareness_moore.set_index([0,0,0], 0)
stencil_squareness_t = np.transpose(stencil_squareness_moore) 

# creating neighborhood definition 
stencil_squareness_von = tg.create_stencil("von_neumann", 1, 1)
# Reshaping the moore neighbourhood
stencil_squareness_von[0,:,:] = 0 
stencil_squareness_von[2,:,:] = 0
stencil_squareness_von.set_index([0,0,0], 0)