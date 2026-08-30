from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import WineData, WineDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the wine-related attributes expected from the form.
    """

    def __init__(self, request: Request):
        self.request: Request = request

        self.fixed_acidity: Optional[float] = None
        self.volatile_acidity: Optional[float] = None
        self.citric_acid: Optional[float] = None
        self.residual_sugar: Optional[float] = None
        self.chlorides: Optional[float] = None
        self.free_sulfur_dioxide: Optional[float] = None
        self.total_sulfur_dioxide: Optional[float] = None
        self.density: Optional[float] = None
        self.pH: Optional[float] = None
        self.sulphates: Optional[float] = None
        self.alcohol: Optional[float] = None

    async def get_wine_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """

        form = await self.request.form()

        self.fixed_acidity = form.get("fixed_acidity")
        self.volatile_acidity = form.get("volatile_acidity")
        self.citric_acid = form.get("citric_acid")
        self.residual_sugar = form.get("residual_sugar")
        self.chlorides = form.get("chlorides")
        self.free_sulfur_dioxide = form.get("free_sulfur_dioxide")
        self.total_sulfur_dioxide = form.get("total_sulfur_dioxide")
        self.density = form.get("density")
        self.pH = form.get("pH")
        self.sulphates = form.get("sulphates")
        self.alcohol = form.get("alcohol")


# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for wine quality data input.
    """

    return templates.TemplateResponse(
        "winedata.html",
        {
            "request": request,
            "context": "Rendering"
        }
    )


# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """

    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """

    try:

        form = DataForm(request)

        await form.get_wine_data()

        wine_data = WineData(
            fixed_acidity=form.fixed_acidity,
            volatile_acidity=form.volatile_acidity,
            citric_acid=form.citric_acid,
            residual_sugar=form.residual_sugar,
            chlorides=form.chlorides,
            free_sulfur_dioxide=form.free_sulfur_dioxide,
            total_sulfur_dioxide=form.total_sulfur_dioxide,
            density=form.density,
            pH=form.pH,
            sulphates=form.sulphates,
            alcohol=form.alcohol
        )

        # Convert form data into a DataFrame for the model
        wine_df = wine_data.get_wine_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = WineDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=wine_df)[0]

        # Display prediction result
        status = f"Wine Quality: {value}"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "winedata.html",
            {
                "request": request,
                "context": status
            }
        )

    except Exception as e:

        return {
            "status": False,
            "error": f"{e}"
        }


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )
