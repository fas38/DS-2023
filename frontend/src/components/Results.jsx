import React from "react";
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

function Results(props) {
  let results = Object.values(props.results);

  let actual = results[0];
  let predicted = results[1];

  let normalizedPredicted;
  let values = [];

 // This block of code is used to normalize the predicted values, check of nulls
 // Also the circular progress bar is showing the averaged values
 // the values list has all the values for the line chart
  if(!predicted) {
    normalizedPredicted = 0;
  } else {
    let sum = predicted.reduce((acc, curr) => acc + curr, 0);
    normalizedPredicted = (sum/predicted.length) * 100;
    values = Object.entries(predicted)
  }

  // Prepare data for the line chart
  const chartData = values.map(([key, value]) => ({ name: key, value }));

  return (
    <div>
      <h1>Results</h1>
      <div style={{ width: '300px', height: '300px', margin: '0 auto' }}>
        <CircularProgressbar 
          value={normalizedPredicted} 
          text={`${normalizedPredicted.toFixed(2)}%`} 
          styles={buildStyles({
            textColor: "#000000",
            pathColor: "#22d625",
            trailColor: "#de3a3a",
          })}
        />
      </div>

      <div style={{ width: '500px', height: '300px', margin: '0 auto', paddingTop: '50px' }}>
        <LineChart width={500} height={300} data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#22d625" activeDot={{ r: 8 }} />
        </LineChart>
      </div>

    </div>
  )

}

export default Results;
