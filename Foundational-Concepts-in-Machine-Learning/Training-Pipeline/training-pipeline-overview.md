# Training pipeline

A training pipeline needs to handle a large volume of data with low costs. One common solution is to store data in a column-oriented format like Parquet or ORC. These data formats enable high throughput for ML and analytics use cases. In other use cases, the tfrecord data format is widely used in the TensorFlow ecosystem.

```python3
import pandas as pd

# Create a sample DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Write the DataFrame to a Parquet file
df.to_parquet('sample.parquet')

# Read the Parquet file into a DataFrame
df_read = pd.read_parquet('sample.parquet')

print(df_read)
```

Print Result
```
      name  age         city
0    Alice   25     New York
1      Bob   30  Los Angeles
2  Charlie   35      Chicago
```
