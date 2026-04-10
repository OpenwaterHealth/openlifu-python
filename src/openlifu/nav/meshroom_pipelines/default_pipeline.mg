{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "DepthMap": "5.0",
            "DepthMapFilter": "4.0",
            "FeatureExtraction": "1.3",
            "FeatureMatching": "2.0",
            "MeshFiltering": "3.0",
            "Meshing": "7.0",
            "PrepareDenseScene": "3.1",
            "Publish": "1.3",
            "StructureFromMotion": "3.3",
            "Texturing": "6.0"
        }
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [0, 0],
            "inputs": {}
        },
        "FeatureExtraction_1": {
            "nodeType": "FeatureExtraction",
            "position": [200, 0],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "describerTypes": ["dspsift"],
                "forceCpuExtraction": false
            }
        },
        "FeatureMatching_1": {
            "nodeType": "FeatureMatching",
            "position": [400, 0],
            "inputs": {
                "input": "{FeatureExtraction_1.input}",
                "featuresFolders": ["{FeatureExtraction_1.output}"],
                "imagePairsList": "",
                "describerTypes": "{FeatureExtraction_1.describerTypes}"
            }
        },
        "StructureFromMotion_1": {
            "nodeType": "StructureFromMotion",
            "position": [600, 0],
            "inputs": {
                "input": "{FeatureMatching_1.input}",
                "featuresFolders": "{FeatureMatching_1.featuresFolders}",
                "matchesFolders": ["{FeatureMatching_1.output}"],
                "describerTypes": "{FeatureMatching_1.describerTypes}"
            }
        },
        "PrepareDenseScene_1": {
            "nodeType": "PrepareDenseScene",
            "position": [800, 0],
            "inputs": {
                "input": "{StructureFromMotion_1.output}"
            }
        },
        "DepthMap_1": {
            "nodeType": "DepthMap",
            "position": [1000, 0],
            "inputs": {
                "input": "{PrepareDenseScene_1.input}",
                "imagesFolder": "{PrepareDenseScene_1.output}",
                "downscale": 2
            }
        },
        "DepthMapFilter_1": {
            "nodeType": "DepthMapFilter",
            "position": [1200, 0],
            "inputs": {
                "input": "{DepthMap_1.input}",
                "depthMapsFolder": "{DepthMap_1.output}"
            }
        },
        "Meshing_1": {
            "nodeType": "Meshing",
            "position": [1400, 0],
            "inputs": {
                "input": "{DepthMapFilter_1.input}",
                "depthMapsFolder": "{DepthMapFilter_1.output}"
            }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "position": [1600, 0],
            "inputs": {
                "inputMesh": "{Meshing_1.outputMesh}"
            }
        },
        "Texturing_1": {
            "nodeType": "Texturing",
            "position": [1800, 0],
            "inputs": {
                "input": "{Meshing_1.output}",
                "imagesFolder": "{DepthMap_1.imagesFolder}",
                "inputMesh": "{MeshFiltering_1.outputMesh}",
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                }
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [2000, 0],
            "inputs": {
                "inputFiles": [
                    "{Texturing_1.output}",
                    "{Texturing_1.outputMesh}",
                    "{Texturing_1.outputMaterial}",
                    "{Texturing_1.outputTextures}"
                ]
            }
        }
    }
}
