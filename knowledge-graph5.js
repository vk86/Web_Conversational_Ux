import React, { useState, useCallback, useEffect } from 'react';
import ReactFlow, { addEdge, MiniMap, Controls, Background } from 'react-flow-renderer';
import ResizablePosterBox from './ResizablePosterBox';  // Updated import
import './Styles.css'; // Add separate CSS file

const KnowledgeGraph = ({ userStoryId }) => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [expandedNodes, setExpandedNodes] = useState(new Set());
  const [storyData, setStoryData] = useState(null);

  // Mock Data
  const data = {
    "_id": "user_story_1",
    "type": "UserStory",
    "title": "Implement login functionality",
    "description": "As a user, I want to log in to the system using my credentials.",
    "test_cases": [
      {
        "test_case_id": "test_case_1",
        "title": "Verify login with valid credentials",
        "description": "Ensure that users can log in with valid credentials.",
        "test_scripts": [
          {
            "test_script_id": "test_script_1",
            "name": "LoginScript.py",
            "content": "Selenium-based script for testing login functionality."
          }
        ],
        "test_data": [
          {
            "test_data_id": "test_data_1",
            "data": {
              "username": "user@example.com",
              "password": "password123"
            }
          }
        ]
      },
      {
        "test_case_id": "test_case_2",
        "title": "Verify login with invalid credentials",
        "description": "Ensure that users cannot log in with invalid credentials.",
        "test_scripts": [
          {
            "test_script_id": "test_script_2",
            "name": "InvalidLoginScript.py",
            "content": "Selenium-based script for testing invalid login attempts."
          }
        ],
        "test_data": [
          {
            "test_data_id": "test_data_2",
            "data": {
              "username": "invalid_user@example.com",
              "password": "wrong_password"
            }
          }
        ]
      }
    ]
  };

  // Fetch user story data
  useEffect(() => {
    const fetchUserStoryData = async () => {
      try {
        setStoryData(data);
        const initialNodes = [
          {
            id: data._id,
            data: {
              label: (
                <div className="node-wrapper">
                  <div className="node-title">User Story</div> {/* Add title */}
                  <ResizablePosterBox
                    id={`User Story ID: ${data._id}`}
                    title={data.title}
                    description={data.description}
                  />
                </div>
              ),
            },
            position: { x: 100, y: 100 },  // Starting position
            type: 'default',
            draggable: true
          },
        ];
        setNodes(initialNodes);
      } catch (error) {
        console.error('Error fetching user story data:', error);
      }
    };
    fetchUserStoryData();
  }, [userStoryId]);

  const expandNode = (nodeId, level, nodeData = null) => {
    if (expandedNodes.has(nodeId)) {
      // Collapse the node by removing its children
      setNodes((nds) => nds.filter((n) => !n.id.startsWith(`${nodeId}-`)));
      setEdges((eds) => eds.filter((e) => !e.source.startsWith(`${nodeId}-`)));
      setExpandedNodes((prev) => {
        const newExpanded = new Set(prev);
        newExpanded.delete(nodeId);
        return newExpanded;
      });
    } else {
      let newNodes = [];
      let newEdges = [];

      if (level === 'userStory') {
        const testCases = nodeData.test_cases || [];
        testCases.forEach((testCase, index) => {
          const testCaseNodeId = `${nodeId}-testcase-${index}`;
          newNodes.push({
            id: testCaseNodeId,
            data: {
              label: (
                <div className="node-wrapper">
                  <div className="node-title">Test Case</div> {/* Add title */}
                  <ResizablePosterBox
                    id={testCase.test_case_id}
                    title={testCase.title}
                    description={testCase.description}
                  />
                </div>
              ),
            },
            position: { x: 400, y: 300 * index },
            type: 'default',
            draggable: true
          });

          newEdges.push({
            id: `e${nodeId}-testcase-${index}`,
            source: nodeId,
            target: testCaseNodeId,
            type: 'smoothstep',
            markerEnd: 'url(#arrow)',
            sourcePosition: 'right',  // Start from the right of Test Case node
            targetPosition: 'left'
            });

          // Test Scripts and Test Data here...
          const testScripts = testCase.test_scripts || [];
          testScripts.forEach((script, scriptIndex) => {
            const testScriptNodeId = `${testCaseNodeId}-testscript-${scriptIndex}`;
            newNodes.push({
              id: testScriptNodeId,
              data: {
                label: (
                  <div className="node-wrapper">
                    <div className="node-title">Test Script</div> {/* Add title */}
                    <ResizablePosterBox
                      id={script.test_script_id}
                      title={script.name}
                      description={script.content}
                    />
                  </div>
                ),
              },
              position: { x: 700, y: 300 * index + 150 * scriptIndex },
              type: 'default',
              draggable: true
            });

            newEdges.push({
              id: `e${testCaseNodeId}-testscript-${scriptIndex}`,
              source: testCaseNodeId,
              target: testScriptNodeId,
              type: 'smoothstep',
              markerEnd: 'url(#arrow)'
            });

            // Test Data for each Test Script
            const testData = testCase.test_data || [];
            testData.forEach((dataItem, dataIndex) => {
              const testDataNodeId = `${testScriptNodeId}-testdata-${dataIndex}`;
              
              // Add Test Data Node
              newNodes.push({
                id: testDataNodeId,
                data: {
                  label: (
                    <div className="node-wrapper">
                      <div className="node-title">Test Data</div> {/* Add title for Test Data */}
                      <ResizablePosterBox
                        id={dataItem.test_data_id}
                        title="Test Data"
                        description={`Username: ${dataItem.data.username}, Password: ${dataItem.data.password}`}
                      />
                    </div>
                  ),
                },
                position: { x: 1000, y: 300 * index + 150 * scriptIndex + 100 * dataIndex },  // Adjusted position for Test Data
                type: 'default',
                draggable: true
              });

              // Add Edge between Test Script and Test Data
              newEdges.push({
                id: `e${testScriptNodeId}-testdata-${dataIndex}`,
                source: testScriptNodeId,
                target: testDataNodeId,
                type: 'smoothstep',
                markerEnd: 'url(#arrow)'
              });
            });
          });
        });
      }

      setNodes((nds) => nds.concat(newNodes));
      setEdges((eds) => eds.concat(newEdges));
      setExpandedNodes((prev) => new Set(prev).add(nodeId));
    }
  };

  const onNodeClick = (event, node) => {
    if (storyData && node.id === storyData._id) {
      expandNode(node.id, 'userStory', storyData);
    }

    if (node.id.includes('testcase')) {
      const parentIndex = node.id.split('-').pop();
      const selectedTestCase = storyData.test_cases[parentIndex];
      expandNode(node.id, 'testCase', selectedTestCase);
    }
  };

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    []
  );

  return (
    <div className="reactflow-wrapper" style={{ height: '100vh', width: '100vw' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={onNodeClick}
        onConnect={onConnect}
        fitView
      >
        <svg>
          <defs>
            <marker
              id="arrow"
              markerWidth="10"
              markerHeight="10"
              refX="10"
              refY="5"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L0,10 L10,5 z" fill="#000" />
            </marker>
          </defs>
        </svg>
        <MiniMap nodeColor={(node) => (node.type === 'input' ? 'blue' : '#FFCC00')} />
        <Controls />
        <Background color="#aaa" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default KnowledgeGraph;