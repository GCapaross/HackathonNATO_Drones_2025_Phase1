# NATO IST Hackathon - Phase 1: AI-based Drone Detection and Classification

## Overview
This hackathon focuses on developing AI solutions for drone detection and classification using RF signals. The project is divided into two main components, each worth 10 points (50% of total score).

## Phase 1 Timeline: September 19-30, 2025
- **Kick-off**: September 19, 2025 (11h) at DETI-UA
- **Q&A Session**: September 24, 2025 (17h) at IT2 building
- **Evaluation**: October 1, 2025
- **Pre-selection**: Top teams advance to Phase 2

## Project Components

### 1. Automatic Data Labelling (10 points - 50% of score)

**Objective**: Develop automated systems for labeling RF signal data with minimal human intervention.

**Evaluation Criteria**:
- **Automation Quality (4 pts)**: From mostly manual (0) to highly automated with minimal errors (4)
- **Human Verification Integration (2 pts)**: From no verification loop (0) to efficient human-in-the-loop with minimal effort (2)
- **Accuracy of Labels (2 pts)**: From inconsistent/error-prone (0-1) to reliable and consistent labels (2)
- **Innovation & Usability (2 pts)**: From standard/basic (0-1) to novel, efficient, user-friendly (2)

**Key Requirements**:
- Minimize manual labeling effort
- Implement human-in-the-loop verification for uncertain cases
- Focus on YOLO format annotations
- Create user-friendly GUI for verification
- Automate repetitive labeling tasks

### 2. Detection & Classification (10 points - 50% of score)

**Objective**: Develop AI models for accurate drone detection and classification from RF signals.

**Evaluation Criteria**:
- **Accuracy (4 pts)**: From poor (<60%) (0-1) to high (>85%) with generalization (4)
- **Latency & Efficiency (2 pts)**: From heavy models, high inference (0-1) to optimized real-time models (2)
- **Robustness & Scalability (2 pts)**: From unstable/limited (0-1) to stable across datasets, scalable (2)
- **Innovation & Explainability (2 pts)**: From standard pipeline (0-1) to innovative and/or interpretable (2)

**Key Requirements**:
- Achieve >85% accuracy with good generalization
- Optimize for real-time performance
- Ensure robustness across different datasets
- Implement explainable AI approaches

## Available Resources

### Datasets
- **CSV Files**: RF signal data (100GB+)
- **Spectrogram Training Data**: 23,000+ PNG images and text files
- **Multiple Bandwidths**: 25MHz, 45MHz, 60MHz, 125MHz
- **Signal Types**: BLE, Bluetooth Classic, WLAN, etc.

### Code Implementations
1. **DroneRF**: Deep learning approaches for RF-based drone detection
2. **MDPI Dataset Helper**: Scripts for spectrogram generation and YOLO/COCO format conversion
3. **S3R**: Open Set Learning for RF-based drone recognition

### Technical Articles
- Combined RF-Based Drone Detection and Classification
- Deep Learning for RF Fingerprinting
- Enhancing UAV Network Security
- Open Set Learning for RF-Based Drone Recognition
- And more...

## Development Strategy

### Week 1-2: Data Labelling Pipeline
- Set up YOLO annotation pipeline
- Create automated labeling scripts
- Implement human verification GUI
- Test on sample datasets

### Week 3-4: Classification Models
- Implement CNN for spectrogram classification
- Test existing DroneRF and S3R approaches
- Develop hybrid traditional + AI methods
- Optimize for real-time performance

### Week 5-6: Integration & Optimization
- Combine labelling and classification
- Performance optimization
- Documentation and testing
- Prepare for Phase 2

## Technical Approaches

### Data Labelling
- **YOLO Format**: For object detection and classification
- **Human-in-the-Loop**: GUI for uncertain cases
- **Automation**: Reduce repetitive manual work
- **Quality Control**: Verification and validation systems

### Classification Methods
- **CNN**: For spectrogram analysis
- **Hybrid Approaches**: Traditional methods + AI
- **Multi-Network Architecture**: UAV detection → classification
- **LSTM**: For temporal pattern recognition
- **Ensemble Methods**: Multiple model combinations

## Strategic Approach & Key Insights

### Data Labelling Strategy
- **Real-time Focus**: Don't waste time on manual labeling - focus on automation
- **Human-in-the-Loop**: Use GUI for uncertain cases when confidence is low
- **YOLO Format**: Recommended approach for annotations
- **Progressive Automation**: Start with manual, move to automated labeling
- **AI Training**: Train models on repetitive tasks with variations

### Technical Architecture
- **Hybrid Solutions**: Combine traditional methods (non-AI) with AI approaches
- **Multi-Network Architecture**: 
  - Primary network: UAV detection (yes/no)
  - Secondary networks: Classification based on detection result
- **Preprocessing**: Subdivide images to reduce network complexity
- **Multiple Representations**: Combine time domain and frequency domain signals

### Network Types to Explore
- **ANN, DNN, CNN, RNN**: Test different architectures
- **LSTM**: For temporal pattern recognition
- **Traditional Methods**: Energy-based classification using signal amplitude
- **Architecture Optimization**: Vary number of layers and document results

### Performance Enhancement
- **Data Artifacts**: Use artifacts to improve network performance
- **Preprocessing Algorithms**: Reduce complexity before network processing
- **Ensemble Methods**: Multiple networks for different tasks

## Success Factors

1. **Documentation**: Document everything - it has value
2. **Automation**: Minimize manual work
3. **Real-time Performance**: Models must work in real-time
4. **Robustness**: Solutions should work across different datasets
5. **Innovation**: Novel approaches are valued
6. **Explainability**: Interpretable AI solutions
7. **Exploration**: Test different scenarios and document results
8. **Error Analysis**: Understand why things fail and have backup strategies

## Competitive Advantage

### Current Situation
- **Many teams struggle**: Some haven't even managed to read the labels yet
- **Aerospace background**: Many participants lack AI knowledge
- **Opportunity**: Focus on practical, working solutions rather than perfect accuracy

### Key Differentiators
- **Real-time Focus**: Prioritize working systems over theoretical perfection
- **Hybrid Approach**: Combine traditional signal processing with AI
- **Documentation**: Thorough documentation adds value to the work
- **Robustness**: Build systems that work across different scenarios

## Project Timeline & Deliverables

### Phase 1 Deliverables
- **Light Description**: Brief description of work completed
- **Presentation**: Final presentation on Wednesday
- **Jury Evaluation**: 5-10 teams will advance to Phase 2
- **Documentation**: Detailed documentation of approach and results

### Phase 2 (If Selected)
- **Jinwoo Support**: Additional team member joins
- **NATO Data**: Access to additional datasets and scripts
- **Field Testing**: Real-world deployment and testing

### Awards
- **3 Diplomas**: One for each phase
- **Prizes**: Additional awards for top performers

## Next Steps

1. **Environment Setup**: Install required packages
2. **Data Exploration**: Examine CSV files and understand data format
3. **Code Testing**: Run existing DroneRF and S3R implementations
4. **Tool Development**: Create basic labelling and classification tools
5. **Integration**: Combine components into working system
6. **Documentation**: Start documenting everything from day one

## Support Resources

- **Mentorship**: Teams supported by Telecommunications Institute experts
- **Q&A Session**: September 24, 2025 (17h) at IT2 building
- **Slack Group**: https://join.slack.com/t/natoist/shared_invite/zt-3dn7h43py-_h8388EOI0qA9~jSkcUhVw
- **Team Registration**: https://forms.gle/1fNTqZSg571FaX9r6

## Important Notes

- Teams: 1-3 members
- Registration required for @ua.pt accounts
- Focus on practical feasibility
- Innovation and performance trade-offs are valued
- Documentation is crucial for evaluation

---

**Remember**: The goal is to create a robust, automated system that can detect and classify drones in real-time with minimal human intervention, while maintaining high accuracy and providing explainable results.
