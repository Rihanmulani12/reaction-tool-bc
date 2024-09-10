from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from skopt import gp_minimize
from skopt.space import Real

app = FastAPI()

# Configure CORS to allow requests from http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # No trailing slash
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the input data model
class ReactionParams(BaseModel):
    temperature: float
    pressure: float
    concentration: float
    ph: float
    catalyst: float

# Objective function for Bayesian optimization
def objective_function(params):
    temperature, pressure, concentration, ph, catalyst = params
    yield_value = -(temperature - 70) ** 2 - (pressure - 50) ** 2 - (concentration - 1) ** 2 - (ph - 7) ** 2 - (catalyst - 0.1) ** 2
    return yield_value

# Search space for the optimization
space = [
    Real(50, 100, name="temperature"),
    Real(10, 100, name="pressure"),
    Real(0.1, 2.0, name="concentration"),
    Real(1, 14, name="ph"),
    Real(0.01, 1.0, name="catalyst")
]

# Store intermediate results during optimization
intermediate_results = []

# Callback function to track optimization progress
def track_progress(res):
    intermediate_results.append({
        "iteration": len(intermediate_results) + 1,
        "temperature": res.x_iters[-1][0],
        "pressure": res.x_iters[-1][1],
        "concentration": res.x_iters[-1][2],
        "ph": res.x_iters[-1][3],
        "catalyst": res.x_iters[-1][4],
        "yield": -res.func_vals[-1]  # Track the negative of the objective function
    })

@app.post("/")
def optimize_reaction(params: ReactionParams):
    # Clear intermediate results from previous runs
    global intermediate_results
    intermediate_results = []

    # Run the optimization with the tracking callback
    result = gp_minimize(objective_function, space, n_calls=10, random_state=42, callback=[track_progress])

    # Return the intermediate results and the final optimized values
    return {
        "optimized_temperature": result.x[0],
        "optimized_pressure": result.x[1],
        "optimized_concentration": result.x[2],
        "optimized_ph": result.x[3],
        "optimized_catalyst": result.x[4],
        "optimal_yield": -result.fun,
        "iterations": intermediate_results
    }
