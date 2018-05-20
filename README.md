# Energy-Retrofit-Index
This MATLAB script containes the original implementation of the Energy Retrofit Index as in "https://doi.org/10.1016/j.apenergy.2017.08.237".
The script is tailored to a database with ~5000 samples and 5 features, therefore, users are advised to fine-tune the hyper-parameters based on their specific dataset.
The input dataset should be called "X", with samples in rows and features as columns (e.g. [5000,5]).
The target dataset should be called "T", with samples in rows and ONE column (e.g. [5000,1]).
ERI is the final output of the script in a float format.
Larger ERI values indicate higher retrofit potential.
