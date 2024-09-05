# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Expose port 80 (optional, based on your use case)
EXPOSE 80

# Set the entrypoint to bash so that you can log in interactively
CMD ["/bin/bash"]
