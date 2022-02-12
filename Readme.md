# Exercise for hotel cancellation predictor serving

### Prerequisite:
1. Model training is completed


### Part 1, serve the static webpage:
1. Install Nginx on a VM
2. Create an nginx config so that nginx will know what to serve when a request enters
3. The the config into correct place for nginx to read
4. Restart nginx, and verify the webpage is serving in a browser

### Part 2, serve the Flask app:
1. Finish the flask app
2. Bring up the flask app (with `screen` or `nohup` for now)
3. Config the nginx config such that a request with special path `/api/get_prediction_result` is proxied to the flask app
4. Restart nginx, and verify the webpage got data from the Flask app

### Part 3, Play with Docker