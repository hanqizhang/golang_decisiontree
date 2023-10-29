package lib

import "fmt"

type DecisionTreeNode struct {
	Features []DimensionFilter
	Children []*DecisionTreeNode
}

func (node *DecisionTreeNode) appendChild(child *DecisionTreeNode) {
	node.Children = append(node.Children, child)
}

func (node *DecisionTreeNode) isLeaf() bool {
	return len(node.Children) == 0
}

func (node *DecisionTreeNode) Dump(indent string, output *string, depth int) {
	featureString := ""
	for _, feature := range node.Features {
		featureString += fmt.Sprintf("{Index: %d, value: %s},", feature.ColumnIndex, feature.DimensionValue)
	}
	*output += (indent + "value: " + featureString + "\n")
	if depth < 1 || node.isLeaf() {
		return
	}
	*output += (indent + "children: \n")
	for _, child := range node.Children {
		child.Dump(indent+"\t", output, depth-1)
	}
}

func getSchemaWidth(datasetPath string) int {
	dataset := NewDataset(datasetPath)
	// Exclude target feature
	record, success := dataset.Next([]DimensionFilter{})
	if success {
		// fmt.Printf("%d features:", len(record))
		// for _, val := range record {
		// 	fmt.Printf("%s, ", val)
		// }
		// fmt.Println()
		return len(record)
	} else {
		panic("Empty or malformed file!")
	}
}

func getRemainingFeatures(features []DimensionFilter, numFeatures int) []int {
	// Each entry n maps to the n+1-th column index.
	exists := make([]bool, numFeatures)
	for _, feature := range features {
		exists[feature.ColumnIndex-1] = true
	}

	remainingFeatures := make([]int, 0)
	for index := 1; index <= numFeatures; index++ {
		if !exists[index-1] {
			remainingFeatures = append(remainingFeatures, index)
		}
	}

	return remainingFeatures
}

type DecisionTreeBuilderSeq struct {
	datasetPath   string
	dataset       *MaterializedDataset
	root          *DecisionTreeNode
	currentLeaves []*DecisionTreeNode
}

func NewSequentialDecisionTreeBuilder(datasetPath string) *DecisionTreeBuilderSeq {
	builder := new(DecisionTreeBuilderSeq)
	builder.datasetPath = datasetPath
	builder.dataset = NewMaterializedDataset(builder.datasetPath)
	builder.root = new(DecisionTreeNode)
	builder.root.Features = make([]DimensionFilter, 0)
	builder.currentLeaves = make([]*DecisionTreeNode, 0)
	builder.currentLeaves = append(builder.currentLeaves, builder.root)
	return builder
}

func (builder *DecisionTreeBuilderSeq) growTree(numFeatures int) {
	newLeaves := make([]*DecisionTreeNode, 0)
	// fmt.Printf("currently having %d leaves", len(builder.currentLeaves))
	for _, leaf := range builder.currentLeaves {
		// Always nonempty due to how we loop through this method from the outside.
		remainingFeatures := getRemainingFeatures(leaf.Features, numFeatures)
		var bestBreakdown DimensionBreakdown
		var bestDimension int

		bestEntropy := 1.0
		for _, index := range remainingFeatures {
			breakdown := BreakDownSubtreeDimension(builder.dataset, leaf.Features, index)
			entropy := computeEntropy(breakdown)
			if entropy < bestEntropy {
				bestEntropy = entropy
				bestBreakdown = breakdown
				bestDimension = index
			}
		}

		for value := range bestBreakdown.Distribution {
			node := new(DecisionTreeNode)
			node.Features = append(leaf.Features, DimensionFilter{ColumnIndex: bestDimension, DimensionValue: value})
			newLeaves = append(newLeaves, node)
			leaf.appendChild(node)
		}
	}
	builder.currentLeaves = newLeaves
}

func (builder *DecisionTreeBuilderSeq) Build() *DecisionTreeNode {
	numFeatures := getSchemaWidth(builder.datasetPath) - 1
	for i := 0; i < numFeatures; i++ {
		builder.growTree(numFeatures)
	}
	return builder.root
}

type AggregatorTask struct {
	builder     *DecisionTreeBuilderParallel
	leaf        *DecisionTreeNode
	numFeatures int
	Children    []*DecisionTreeNode
}

func (task *AggregatorTask) execute() {
	// Always nonempty due to how we loop through this method from the outside.
	remainingFeatures := getRemainingFeatures(task.leaf.Features, task.numFeatures)
	futures := make([]*Future, len(remainingFeatures))
	for i, index := range remainingFeatures {
		futures[i] = task.builder.AddEvalTask(task.leaf.Features, index)
	}

	var bestBreakdown DimensionBreakdown
	var bestDimension int

	bestEntropy := 1.0
	for _, future := range futures {
		future.Wait()
		evalTask := future.task.(*EvalTask)
		breakdown := evalTask.DimensionBreakdown
		entropy := evalTask.Entropy
		if entropy < bestEntropy {
			bestEntropy = entropy
			bestBreakdown = breakdown
			bestDimension = evalTask.FeatureDimension
		}
	}

	for value := range bestBreakdown.Distribution {
		node := new(DecisionTreeNode)
		node.Features = append(task.leaf.Features, DimensionFilter{ColumnIndex: bestDimension, DimensionValue: value})
		task.Children = append(task.Children, node)
		task.leaf.appendChild(node)
	}
}

type DecisionTreeBuilderParallel struct {
	datasetPath          string
	dataset              *MaterializedDataset
	evalWorkerPool       *WorkerPool
	aggregatorWorkerPool *WorkerPool
	root                 *DecisionTreeNode
	currentLeaves        []*DecisionTreeNode
}

func NewParallelDecisionTreeBuilder(datasetPath string, evalWorkerPool *WorkerPool, aggregatorWorkerPool *WorkerPool) *DecisionTreeBuilderParallel {
	builder := new(DecisionTreeBuilderParallel)
	builder.datasetPath = datasetPath
	builder.dataset = NewMaterializedDataset(builder.datasetPath)
	// Decouple eval and aggregator worker pools because aggregation tasks will have to
	// wait for eval tasks, and running a shared pool risks starvation of eval tasks,
	// which consequently leads to program stall.
	builder.evalWorkerPool = evalWorkerPool
	builder.aggregatorWorkerPool = aggregatorWorkerPool
	builder.root = new(DecisionTreeNode)
	builder.root.Features = make([]DimensionFilter, 0)
	builder.currentLeaves = make([]*DecisionTreeNode, 0)
	builder.currentLeaves = append(builder.currentLeaves, builder.root)
	return builder
}

func (builder *DecisionTreeBuilderParallel) AddEvalTask(
	subtreeDesc []DimensionFilter,
	featureDimension int) *Future {
	task := new(EvalTask)
	task.Dataset = builder.dataset
	task.SubtreeDesc = subtreeDesc
	task.FeatureDimension = featureDimension
	future := NewFuture(task)
	builder.evalWorkerPool.AddTask(future)
	return future
}

func (build *DecisionTreeBuilderParallel) AddAggregatorTask(
	builder *DecisionTreeBuilderParallel,
	leaf *DecisionTreeNode,
	numFeatures int) *Future {
	task := new(AggregatorTask)
	task.builder = builder
	task.leaf = leaf
	task.numFeatures = numFeatures
	task.Children = make([]*DecisionTreeNode, 0)
	future := NewFuture(task)
	builder.aggregatorWorkerPool.AddTask(future)
	return future
}

func (builder *DecisionTreeBuilderParallel) growTree(numFeatures int) {
	currentLeafCount := len(builder.currentLeaves)
	newLeaves := make([]*DecisionTreeNode, 0)
	futures := make([]*Future, currentLeafCount)
	for i, leaf := range builder.currentLeaves {
		futures[i] = builder.AddAggregatorTask(builder, leaf, numFeatures)
	}

	for _, future := range futures {
		future.Wait()
		aggregatorTask := future.task.(*AggregatorTask)
		for _, leaf := range aggregatorTask.Children {
			newLeaves = append(newLeaves, leaf)
		}
	}
	builder.currentLeaves = newLeaves
}

func (builder *DecisionTreeBuilderParallel) Build() *DecisionTreeNode {
	numFeatures := getSchemaWidth(builder.datasetPath) - 1
	for i := 0; i < numFeatures; i++ {
		builder.growTree(numFeatures)
	}
	return builder.root
}
