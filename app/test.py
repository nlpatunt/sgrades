{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "80db0900",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pandas'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[12]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpandas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpd\u001b[39;00m\n\u001b[32m      2\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mjson\u001b[39;00m\n\u001b[32m      4\u001b[39m \u001b[38;5;66;03m# Function to convert JSONL to CSV\u001b[39;00m\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'pandas'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import json\n",
    "\n",
    "# Function to convert JSONL to CSV\n",
    "def jsonl_to_csv(jsonl_file, csv_file):\n",
    "    # Open the JSONL file and read it line by line\n",
    "    with open(jsonl_file, 'r') as f:\n",
    "        # Load each line as a JSON object and store in a list\n",
    "        data = [json.loads(line) for line in f]\n",
    "    \n",
    "    # Convert the list of JSON objects to a pandas DataFrame\n",
    "    df = pd.DataFrame(data)\n",
    "    \n",
    "    # Save the DataFrame to a CSV file\n",
    "    df.to_csv(csv_file, index=False)\n",
    "\n",
    "# Example usage\n",
    "jsonl_to_csv('input.jsonl', 'output.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2376311",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pandas'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpandas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpd\u001b[39;00m\n\u001b[32m      2\u001b[39m \u001b[38;5;28mprint\u001b[39m(pd.__version__)\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'pandas'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "print(pd.__version__)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
