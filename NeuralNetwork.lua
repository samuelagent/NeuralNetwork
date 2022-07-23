local TrainingData = require(script:WaitForChild("TrainingData"))
local MatrixMath = require(script:WaitForChild("MatrixMath"))

local UserInputService = game:GetService("UserInputService")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

local Player = Players.LocalPlayer
local Camera = workspace.CurrentCamera

local Frame = script.Parent

local ComputeButton = Frame:WaitForChild("Compute")
local ExportButton = Frame:WaitForChild("Export")
local ClearButton = Frame:WaitForChild("Clear")
local TrainButton = Frame:WaitForChild("Train")

local Information = Frame:WaitForChild("Information")
local Canvas = Frame:WaitForChild("Canvas")
local Guess = Frame:WaitForChild("Guess")

-- CONSTANTS

local InputNodes = 784
local OutputNodes = 10

local HiddenNodes = 35 -- Per Hidden Layer
local HiddenLayers = 3

local BatchSize = 4
local LearningRate = 0.0001

-- RUN-TIME TABLES

local Predictions = {
	Correct = 0,
	Incorrect = 0,
	Total = 0
}

local InputData = {} -- To be ported from the MNIST data set, or canvas
InputData = MatrixMath:Random(BatchSize, InputNodes, 0, 1)
local OutputData = {} -- To be ported from the MNIST data set, or canvas
OutputData = MatrixMath:Random(BatchSize, OutputNodes, 0, 1)
local WeightMatrices = {}
local BiasMatrices = {}
if HiddenLayers > 0 then
	table.insert(WeightMatrices, MatrixMath:Random(InputNodes, HiddenNodes, -1, 1))
	table.insert(BiasMatrices, MatrixMath:Random(1, HiddenNodes, -1, 1))
	for count = 1, HiddenLayers - 1 do
		table.insert(WeightMatrices, MatrixMath:Random(HiddenNodes, HiddenNodes, -1, 1))
		table.insert(BiasMatrices, MatrixMath:Random(1, HiddenNodes, -1, 1))
	end
	table.insert(WeightMatrices, MatrixMath:Random(HiddenNodes, OutputNodes, -1, 1))
	table.insert(BiasMatrices, MatrixMath:Random(1, OutputNodes, -1, 1))
else
	table.insert(WeightMatrices, MatrixMath:Random(InputNodes, OutputNodes, -1, 1))
	table.insert(BiasMatrices, MatrixMath:Random(1, OutputNodes, -1, 1))
end

local function FindAverage(Inputs)
	local Sum = 0
	for _, v in ipairs(Inputs) do
		Sum += v
	end
	return Sum / #Inputs
end

local function FindSum(Inputs)
	local Sum = 0
	for _, v in ipairs(Inputs) do
		Sum += v
	end
	return Sum
end

local function DeepCopy(original)
	local copy = {}
	for k, v in pairs(original) do
		if type(v) == "table" then
			v = DeepCopy(v)
		end
		copy[k] = v
	end
	return copy
end

local function Cost(Output, CorrectOutput)
	local OutputCosts = {}
	for i, v in ipairs(Output) do
		table.insert(OutputCosts, (v - if i == CorrectOutput then 1 else 0)^2)
	end
	return FindAverage(OutputCosts)
	--return FindSum(OutputCosts)
end

local function ELU(Input)
	if type(Input) == "table" then
		local Table = {}
		for i, v in ipairs(Input) do
			Table[i] = ELU(v)
		end
		return Table
	end
	if Input >= 0 then return Input end
	return math.exp(Input) - 1
end

local function ProcessLayer(Layer, Weights, LayerIndex)
	local Output = MatrixMath:Dot(Layer, Weights)
	Output = MatrixMath:Add(Output, BiasMatrices[LayerIndex])
	Output = ELU(Output)
	return Output
end

local function ProcessOutput(Output)
	local Thing = math.max(table.unpack(Output))
	return table.find(Output, Thing)
end

local function TableOutput(Output)
	local Table = {}
	for count = 1,10 do
		table.insert(Table, if count == Output then 1 else 0)
	end
	return Table
end

local function ForwardPropogate()
	if HiddenLayers < 1 then
		return ProcessLayer(InputData, WeightMatrices[1])
	end
	
	local LayerOutputs = {}
	table.insert(LayerOutputs, ELU(ProcessLayer(InputData, WeightMatrices[1], 1)))
	
	for count = 1, HiddenLayers do
		table.insert(LayerOutputs, ELU(ProcessLayer(LayerOutputs[#LayerOutputs], WeightMatrices[count + 1], count + 1)))
	end
	return LayerOutputs
end

local function BackPropogate(LayerOutputs) -- # of LayerOutputs is HiddenLayers + 1, same as weights present
	if HiddenLayers < 1 then LayerOutputs = {LayerOutputs} end
	local GradientLayers = {}
	GradientLayers[#LayerOutputs] = MatrixMath:Multiply(2, (MatrixMath:Subtract(LayerOutputs[#LayerOutputs], OutputData)))
	local GradientWeights = {}
	local GradientBiases = {}
	
	for count = #LayerOutputs, 2, -1 do
		GradientWeights[count] = MatrixMath:Dot(MatrixMath:Tranpose(LayerOutputs[count - 1]), GradientLayers[count])
		--GradientBiases[count] = MatrixMath:Add(MatrixMath:Tranpose(LayerOutputs[count - 1], 2))
		
		GradientLayers[count - 1] = ELU(MatrixMath:Dot(GradientLayers[count], MatrixMath:Tranpose(WeightMatrices[count])))
	end
	
	GradientWeights[1] = MatrixMath:Dot(MatrixMath:Tranpose(InputData), GradientLayers[1]) -- Gradietn with respect to w1 weights matrix
	
	for count = 1, #WeightMatrices do
		WeightMatrices[count] = MatrixMath:Subtract(WeightMatrices[count], MatrixMath:Multiply(GradientWeights[count], LearningRate))	
	end
	--print(Cost(LayerOutputs[#LayerOutputs][1], OutputData[1]))
end


local function GetAdjacentCells(Cell)
	local AdjecentCells = {}
	if Cell > 28 then
		table.insert(AdjecentCells, Cell - 28) -- Up
	end
	if Cell < 756 then
		table.insert(AdjecentCells, Cell + 28) -- Down
	end
	if Cell < 784 and Cell % 28 ~= 0 then
		table.insert(AdjecentCells, Cell + 1)  -- Right
	end
	if Cell > 1 and Cell % 28 ~= 1 then
		table.insert(AdjecentCells, Cell - 1)  -- Left
	end
	return AdjecentCells
end

local function GetCanvasInputs()
	local Input = {}
	for Cell = 1, 784 do
		table.insert(Input, 1 - Canvas[Cell].BackgroundTransparency)
	end
	return {Input}
end

for Cell = 1, 784 do
	Canvas[Cell].InputBegan:Connect(function(Input)
		if (Input.UserInputType == Enum.UserInputType.MouseMovement and UserInputService:IsMouseButtonPressed(Enum.UserInputType.MouseButton1)) or Input.UserInputType == Enum.UserInputType.MouseButton1 then
			Canvas[Cell].BackgroundTransparency = 0
			for _, AdjacentCell in pairs(GetAdjacentCells(Cell)) do
				if Canvas[AdjacentCell].BackgroundTransparency > 0 then
					Canvas[AdjacentCell].BackgroundTransparency -= 1/3
				end
			end
		end
	end)
end

local function ClearCanvas()
	for Cell = 1, 784 do
		Canvas[Cell].BackgroundTransparency = 1
	end
	Predictions["Correct"] = 0
	Predictions["Incorrect"] = 0
	Predictions["Total"] = 0
end

local function Train(Times)
	for count = 1, Times do
		local SampleData = TrainingData:GetTrainingSamples(BatchSize)
		
		local NewOutputData = {}
		local NewInputData = {}
		for count = 1, BatchSize do
			table.insert(NewOutputData, TableOutput(SampleData[count][1]))
			table.insert(NewInputData, SampleData[count][2])
		end
	
		OutputData = NewOutputData
		InputData = NewInputData
		
		local LayerOutputs = ForwardPropogate()
		local LayerPredictions = LayerOutputs[#LayerOutputs]
		for i, v in pairs(LayerPredictions) do
			local Prediction = ProcessOutput(v)
			local Answer = ProcessOutput(NewOutputData[i])

			Predictions["Total"] += 1
			if Answer == Prediction then
				Predictions["Correct"] += 1
			else
				Predictions["Incorrect"] += 1
			end
		end
		
		BackPropogate(LayerOutputs)
	end
end

Train(1)

local PreviousBatchSize = BatchSize
local function Compute()
	BatchSize = 1
	InputData = GetCanvasInputs()
	
	local LayerOutputs = ForwardPropogate()
	BatchSize = PreviousBatchSize
end

local function Export()
	print(WeightMatrices)
	print(BiasMatrices)
end

ComputeButton.MouseButton1Click:Connect(Compute)
ClearButton.MouseButton1Click:Connect(ClearCanvas)
ExportButton.MouseButton1Click:Connect(Export)
TrainButton.MouseButton1Click:Connect(function() Train(800) end)
