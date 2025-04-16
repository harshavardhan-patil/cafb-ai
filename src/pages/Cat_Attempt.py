import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { LineChart, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import * as d3 from 'd3';

const TopicModelingVisualization = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('distribution');
  const [processedData, setProcessedData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  
  // Color scale for consistent colors across visualizations
  const colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"];
  
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const fileContent = await window.fs.readFile('jira_issues.csv', { encoding: 'utf8' });
        
        // Parse CSV
        Papa.parse(fileContent, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            setData(results.data);
            const processed = processTopics(results.data);
            setProcessedData(processed);
            setNetworkData(generateNetworkData(processed));
            setLoading(false);
          },
          error: (error) => {
            setError(`Failed to parse CSV: ${error}`);
            setLoading(false);
          }
        });
      } catch (err) {
        setError(`Failed to load file: ${err.message}`);
        setLoading(false);
      }
    };
    
    loadData();
  }, []);
  
  // Define stopwords for text processing
  const stopwords = new Set([
    'the', 'and', 'for', 'this', 'that', 'with', 'was', 'not', 'has', 'are', 'is', 'you', 'have',
    'to', 'in', 'of', 'a', 'on', 'be', 'it', 'by', 'as', 'at', 'an', 'we', 'our', 'us', 'can'
  ]);
  
  // Define topics based on keywords
  const topics = {
    'Technical Issues': ['error', 'bug', 'issue', 'fix', 'problem', 'crash', 'code', 'server', 'failure', 'broken'],
    'UI/UX': ['ui', 'interface', 'design', 'button', 'screen', 'display', 'user', 'layout', 'visual', 'navigation'],
    'Performance': ['slow', 'speed', 'performance', 'fast', 'optimization', 'load', 'latency', 'response', 'timeout'],
    'Feature Requests': ['feature', 'add', 'implement', 'new', 'enhancement', 'request', 'functionality', 'capability'],
    'Documentation': ['doc', 'documentation', 'guide', 'help', 'manual', 'information', 'instruction', 'knowledge'],
    'Security': ['security', 'access', 'permission', 'authentication', 'authorization', 'login', 'password', 'protect'],
    'Integration': ['integration', 'connect', 'api', 'sync', 'plugin', 'module', 'external', 'service']
  };
  
  // Function to tokenize text
  const tokenize = (text) => {
    if (!text) return [];
    return String(text)
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopwords.has(word));
  };
  
  // Function to assign topics to an issue
  const assignTopics = (text) => {
    const words = tokenize(text);
    const scores = {};
    
    Object.entries(topics).forEach(([topic, keywords]) => {
      scores[topic] = words.filter(word => keywords.includes(word)).length;
    });
    
    const topScore = Math.max(...Object.values(scores));
    if (topScore > 0) {
      // Find all topics with the top score
      return Object.entries(scores)
        .filter(([_, score]) => score === topScore)
        .map(([topic]) => topic);
    }
    
    return ['Other'];
  };
  
  // Process topics from the data
  const processTopics = (issuesData) => {
    // Count issues by topic
    const topicCounts = {};
    const wordsByTopic = {};
    const categoryByTopic = {};
    const issuesByTopic = {};
    
    // Initialize
    const allTopics = [...Object.keys(topics), 'Other'];
    allTopics.forEach(topic => {
      topicCounts[topic] = 0;
      wordsByTopic[topic] = {};
      categoryByTopic[topic] = {};
      issuesByTopic[topic] = [];
    });
    
    // Process each issue
    issuesData.forEach(issue => {
      if (!issue.summary) return;
      
      const issueTopics = assignTopics(issue.summary);
      const words = tokenize(issue.summary);
      
      issueTopics.forEach(topic => {
        // Increment topic count
        topicCounts[topic] = (topicCounts[topic] || 0) + 1;
        
        // Add to issues by topic
        issuesByTopic[topic].push(issue);
        
        // Count words for this topic
        words.forEach(word => {
          wordsByTopic[topic][word] = (wordsByTopic[topic][word] || 0) + 1;
        });
        
        // Count categories for this topic
        if (issue.main_category) {
          categoryByTopic[topic][issue.main_category] = 
            (categoryByTopic[topic][issue.main_category] || 0) + 1;
        }
      });
    });
    
    // Get top words for each topic
    const topWordsByTopic = {};
    Object.entries(wordsByTopic).forEach(([topic, words]) => {
      topWordsByTopic[topic] = Object.entries(words)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([word, count]) => ({ text: word, value: count }));
    });
    
    // Create topic-category data for heatmap
    const heatmapData = [];
    Object.entries(categoryByTopic).forEach(([topic, categories]) => {
      Object.entries(categories).forEach(([category, count]) => {
        if (count > 0) {
          heatmapData.push({
            topic,
            category,
            count
          });
        }
      });
    });
    
    // Transform for charts
    const topicDistribution = Object.entries(topicCounts)
      .map(([topic, count]) => ({ topic, count }))
      .sort((a, b) => b.count - a.count);
    
    return {
      topicCounts,
      topicDistribution,
      wordsByTopic,
      topWordsByTopic,
      categoryByTopic,
      heatmapData
    };
  };
  
  // Generate network data for force-directed graph
  const generateNetworkData = (processed) => {
    if (!processed) return null;
    
    const nodes = [];
    const links = [];
    
    // Add topic nodes
    Object.keys(processed.topicCounts).forEach((topic, index) => {
      nodes.push({
        id: topic,
        group: 1,
        size: Math.sqrt(processed.topicCounts[topic] || 1) * 5
      });
    });
    
    // Add keyword nodes and connect to topics
    Object.entries(processed.topWordsByTopic).forEach(([topic, words]) => {
      words.slice(0, 8).forEach(word => {
        const wordId = `word-${word.text}`;
        
        // Check if node already exists
        if (!nodes.find(n => n.id === wordId)) {
          nodes.push({
            id: wordId,
            name: word.text,
            group: 2,
            size: Math.sqrt(word.value) * 3
          });
        }
        
        // Add link
        links.push({
          source: topic,
          target: wordId,
          value: word.value
        });
      });
    });
    
    return { nodes, links };
  };
  
  // D3 Force-Directed Graph
  const ForceGraph = ({ data }) => {
    const containerRef = React.useRef(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
    
    useEffect(() => {
      if (!containerRef.current || !data) return;
      
      // Get container dimensions
      const containerRect = containerRef.current.getBoundingClientRect();
      setDimensions({
        width: containerRect.width,
        height: 600
      });
      
      // Clear any existing SVG
      d3.select(containerRef.current).select("svg").remove();
      
      // Create SVG
      const svg = d3.select(containerRef.current)
        .append("svg")
        .attr("width", containerRect.width)
        .attr("height", 600)
        .attr("viewBox", [0, 0, containerRect.width, 600]);
      
      // Create force simulation
      const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(containerRect.width / 2, 300))
        .force("collision", d3.forceCollide().radius(d => d.size + 10));
      
      // Draw links
      const link = svg.append("g")
        .selectAll("line")
        .data(data.links)
        .join("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.value));
      
      // Node colors by group
      const nodeColor = d => d.group === 1 ? colors[Object.keys(topics).indexOf(d.id) % colors.length] : "#aaa";
      
      // Draw nodes
      const node = svg.append("g")
        .selectAll(".node")
        .data(data.nodes)
        .join("g")
        .attr("class", "node")
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));
      
      // Add circles to nodes
      node.append("circle")
        .attr("r", d => d.size)
        .attr("fill", nodeColor)
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5);
      
      // Add text labels
      node.append("text")
        .text(d => d.id.startsWith('word-') ? d.name : d.id)
        .attr("font-size", d => d.group === 1 ? "14px" : "12px")
        .attr("dx", d => d.size + 5)
        .attr("dy", "0.35em")
        .attr("text-anchor", "start");
      
      // Update positions on simulation tick
      simulation.on("tick", () => {
        link
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);
        
        node
          .attr("transform", d => `translate(${d.x},${d.y})`);
      });
      
      // Drag functions
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
      
      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }
      
      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
      
      return () => {
        simulation.stop();
      };
    }, [data, dimensions]);
    
    return (
      <div ref={containerRef} style={{ width: '100%', height: '600px' }}></div>
    );
  };
  
  // Word cloud for topic terms
  const TopicTerms = ({ topicWords }) => {
    if (!topicWords || topicWords.length === 0) return null;
    
    return (
      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="flex flex-wrap gap-2">
          {topicWords.map((word, idx) => (
            <div 
              key={word.text} 
              className="px-3 py-1 rounded-full"
              style={{ 
                fontSize: '${Math.max(12, Math.min(24, 14 + word.value / 2))}px',
                backgroundColor: 'rgba(70, 130, 180, ${0.3 + word.value / (topicWords[0].value * 2)})' 
              }}
            >
              {word.text}
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  // Topic-Category Matrix visualization
  const TopicCategoryMatrix = ({ data }) => {
    if (!data || !data.heatmapData || data.heatmapData.length === 0) return null;
    
    // Prepare data for the matrix
    const categories = [...new Set(data.heatmapData.map(d => d.category))].slice(0, 10);
    const topics = Object.keys(topics || {}).concat(['Other']);
    
    // Format data for visualization
    const formattedData = [];
    categories.forEach(category => {
      const entry = { category };
      topics.forEach(topic => {
        const match = data.heatmapData.find(d => d.topic === topic && d.category === category);
        entry[topic] = match ? match.count : 0;
      });
      formattedData.push(entry);
    });
    
    return (
      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-4">Topic Distribution Across Categories</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={formattedData}
            margin={{ top: 20, right: 30, left: 20, bottom: 120 }}
            barSize={20}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            {topics.map((topic, index) => (
              <Bar key={topic} dataKey={topic} stackId="a" fill={colors[index % colors.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };
  
  // Topic Distribution Chart
  const TopicDistributionChart = ({ data }) => {
    if (!data || !data.topicDistribution) return null;
    
    return (
      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-4">Topic Distribution</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={data.topicDistribution}
            margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
            barSize={40}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="topic" angle={-45} textAnchor="end" height={80} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8884d8">
              {data.topicDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };
  
  // Render loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-xl font-semibold">Loading and processing data...</div>
      </div>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <div className="p-4 bg-red-100 border border-red-400 text-red-700 rounded mb-4">
        {error}
      </div>
    );
  }
  
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-6">Jira Issues Topic Modeling</h1>
      
      {/* Navigation Tabs */}
      <div className="flex mb-6 border-b">
        <button 
          className={`px-4 py-2 font-medium ${activeTab === 'distribution' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          onClick={() => setActiveTab('distribution')}
        >
          Topic Distribution
        </button>
        <button 
          className={`px-4 py-2 font-medium ${activeTab === 'matrix' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          onClick={() => setActiveTab('matrix')}
        >
          Topic-Category Matrix
        </button>
        <button 
          className={`px-4 py-2 font-medium ${activeTab === 'network' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          onClick={() => setActiveTab('network')}
        >
          Topic Network
        </button>
        <button 
          className={`px-4 py-2 font-medium ${activeTab === 'terms' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          onClick={() => setActiveTab('terms')}
        >
          Topic Terms
        </button>
      </div>
      
      {/* Visualization Content */}
      <div className="mb-6">
        {activeTab === 'distribution' && processedData && (
          <TopicDistributionChart data={processedData} />
        )}
        
        {activeTab === 'matrix' && processedData && (
          <TopicCategoryMatrix data={processedData} />
        )}
        
        {activeTab === 'network' && networkData && (
          <>
            <h3 className="text-lg font-semibold mb-4">Topic-Term Network</h3>
            <div className="bg-white rounded-lg shadow p-2">
              <ForceGraph data={networkData} />
            </div>
          </>
        )}
        
        {activeTab === 'terms' && processedData && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Key Terms by Topic</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(processedData.topWordsByTopic).map(([topic, words]) => (
                <div key={topic} className="bg-white rounded-lg shadow p-4">
                  <h4 className="font-medium mb-2">{topic}</h4>
                  <TopicTerms topicWords={words} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="text-sm text-gray-500 mt-8">
        <p>Note: This visualization uses a keyword-based approach to topic modeling. For more sophisticated topic modeling, algorithms like LDA (Latent Dirichlet Allocation) would provide more accurate results.</p>
      </div>
    </div>
  );
};

export default TopicModelingVisualization;