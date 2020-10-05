

## Running Notebooks from superior.sextonlab.com

1. Start the notebook server

    ./notebooks/scripts/start_jupyter.sh
    # record the security token

Get the `sextonlab_Nvirginia.pem` file and set permissions

   chomd 400 sextonlab_Nvirginia.pem

Start at ssh tunnel:

    ssh -i "sextonlab_Nvirginia.pem" -NfL 8886:localhost:8888 ubuntu@superior.sextonlab.com

open the webbrowser and put in the secret token