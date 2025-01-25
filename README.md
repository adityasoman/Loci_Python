# Loci_Python
Local repo for Loci Python module 

Instllation instructions 

1. Navigate on the conda cmd to the location where you have cloned the Repository
2. Navigate to topogenesis folder 

``` 
cd topogenesis
```
3. Create an enviornment from the win_env.yml file using this command
   ```
   conda env create -f win_env.yml
   ```
4. A new enviornment called locipywinenv will be created. If you want to change this name then you can modify it in the win_env.yml  file
   Activate this new enviornment
   ```
   conda activate locipywinenv
   ```
5. Install Topogenesis by using this command
      ```
   python -m pip install -e .
   ```
