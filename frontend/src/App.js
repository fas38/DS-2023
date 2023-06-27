import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import Results from './components/Results';
import './App.css';

function App() {
  const [searchResults, setSearchResults] = useState([]);

  const handleSearchResults = (data) => {
    setSearchResults(data);
  };

  return (
    <div className="App">
      <SearchBar onSearchResults={handleSearchResults} />
      <Results results={searchResults} />
    </div>
  );
}

export default App;