FROM python:3.12

# Set working directory
WORKDIR /app

# copy source code
COPY . .

# install dependencies with poetry
RUN pip install poetry==1.8.4
# poetry install: This command tells Poetry (a dependency management tool for Python) to install all the 
# dependencies listed in the pyproject.toml file, which is where Poetry manages dependencies for the project.
# --no-dev: This flag tells Poetry to install only the production 
# dependencies (i.e., those needed for running the application) and to exclude any development dependencies. 
# Development dependencies usually include things like testing frameworks (e.g., pytest), 
# linters, or tools required for development but not needed in a production environment.
RUN poetry install --no-dev

# Expose port
EXPOSE 8000

# Run the app
CMD ["poetry", "run", "python", "fastapi_app.py"]

# This Dockerfile is designed to build a Docker image for a Python-based FastAPI application. Here’s a detailed explanation of each line:

#     1. FROM python:3.12
#     Explanation: This defines the base image for the Docker container. The image comes from the official Python image repository, and it's based on Python 3.12.
#     Purpose: This ensures that your container has Python 3.12 installed, which is required to run the Python code in your project.
#     2. WORKDIR /app
#     Explanation: This sets the working directory inside the Docker container to /app. Any subsequent commands, like copying files or installing dependencies, will be executed inside this directory.
#     Purpose: This organizes the container’s file system, ensuring that your project files and dependencies are located in a specific directory (/app).
#     3. COPY . .
#     Explanation: This command copies all files from the current directory on your host machine (the directory where the Dockerfile is located) to the /app directory inside the Docker container.
#     Purpose: This transfers your project code, including the pyproject.toml and fastapi_app.py files, into the container so that they can be accessed and executed.
#     4. RUN pip install poetry==1.8.4
#     Explanation: This uses pip (Python’s package manager) to install a specific version of Poetry, a tool for dependency management and packaging for Python projects. In this case, it's installing version 1.8.4 of Poetry.
#     Purpose: This sets up Poetry so that it can manage and install the project’s dependencies defined in the pyproject.toml file.
#     5. RUN poetry install --no-dev
#     Explanation: This command tells Poetry to install the project’s dependencies as specified in pyproject.toml, but excluding development dependencies, which are often not required in production environments.
#     poetry install: Installs the project’s dependencies.
#     --no-dev: Excludes development dependencies (like testing libraries, linters, or other tools not needed in production).
#     Purpose: Ensures that only necessary production dependencies are installed, which helps optimize the size of the container and minimizes unnecessary packages.
#     6. EXPOSE 8000
#     Explanation: This command declares that the application inside the container will use port 8000. It doesn’t actually open the port on the host machine, but it’s a way of signaling that the container’s app will be running on this port.
#     Purpose: Typically, FastAPI applications use port 8000 by default, so this tells Docker to expose that port for communication when the container runs.
#     7. CMD ["poetry", "run", "python", "fastapi_app.py"]
#     Explanation: This defines the command that will be run when the Docker container starts. It tells Docker to execute:
#     poetry run: This command runs a Python script in an environment where all the dependencies are properly set up by Poetry.
#     python fastapi_app.py: This runs the fastapi_app.py file, which is expected to contain the entry point for your FastAPI web application.
#     Purpose: Starts the FastAPI application when the container is executed.