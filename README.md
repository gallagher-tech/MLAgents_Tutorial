Unity ML Example using ML-Agents package
========================================

What's in this repo
-------------------

This is a guide to getting started with training and running ML agents in Unity, using the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) package. An example project with a premade scene is provided, from which you'll be guided through setup, training, and running (using the [Barracuda inference engine](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Unity-Inference-Engine.md)) an agent.

 - `UnityMLTutorial`: an empty project with the premade scene, within which this tutorial can be carried out by you
 - `UnityMLTutorial_complete`: the completed version of this tutorial for reference

I. Setup
--------

1. Have python 3.6+ installed
2. Create python venv in `UnityMLTutorial` project root dir
3. Decide if you will use CUDA or your CPU ([Verify whether you have a CUDA capable GPU](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system))
3. [Install pytorch python pkg](https://pytorch.org/get-started/locally/) (be sure to select the right Compute Platform corresponding w your CUDA version if using CUDA)
4. Install ML Agents python pkg: `pip install mlagents` (this will install the latest version listed in the latest release [here](https://github.com/Unity-Technologies/ml-agents/releases))
	- Use `mlagents-learn --help` to confirm dl was successful
5. Open UnityMLTutorial project using the latest version of Unity (currently 2020.3.4f1)
6. Install ML-Agents package (Window > Package Manager)
	- NOTE: you need ML-Agents v1.4.0 or higher. If the version in the Package Manager is lower, [follow the instructions here to install the 1.4.0 preview version](https://github.com/Unity-Technologies/ml-agents/issues/4689#issuecomment-735019689).

II. Training
------------

### A. Creating an agent

In the scene there is an **agent** (blue cube) and a goal (yellow sphere) on a platform. The objective is to teach the agent to move toward the goal and not fall off the platform.

1. First we need to create an agent. An agent is the gameobject that's going to be running our AI during both training and inference phases. More specifically, the agent will be a script attached to said gameobject. So create a new script named "MoveToGoalAgent" and open it in an editor
2. Add MLAgents namespace to top of script and inherit from the Agent class instead of Monobehavior:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents; 					/// Add MLAgents namespace

public class MoveToGoalAgent : Agent 	/// Inherit from the Agent class
{
}
```

The way the agent learns is via **reinforcement learning**, which involves looping through the following cycle:

 - **Observation**: gathering data from its environment
 - **Decision**: makes decision based on the data it has
 - **Action**: takes an action in the environment
 - **Reward**: receives a reward if action was correct, this cycle helps the agent discover what action yields the heighest reward

### B. Giving the agent actions

1. Go back to the Unity editor and add your script to the Agent gameobject. In the newly generated **Behavior Parameters component**, which displays the various parameters that our AI uses, set the following params:
 - Behavior Name: `MoveToGoal`; used to reference this behavior in your scripts
 - Vector Action: Space Type: `Discrete`; whether the agent's action(s) are discrete or floating point values
 - Vector Action: Branches Size: `1`; each branch is an action type (i.e., two actions could be "accelerate vs brake" and "turn left vs turn right")
 - Vector Action: Branch 0 Size: `5`; we're making an action with 5 discrete possible outcomes

2. Add the **Decision Requester component** to our agent gameobject (in order to take an action we need to first request a decision). All this script does is request a decision every certain amount of time at which point actions can be taken. Set the following params:
 - Decision Period: `5`; TODO: define **Decision Period**

3. Go back to the MoveAgentToGoal script and add the `OnActionReceived()` method from the `Agent` class:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class MoveToGoalAgent : Agent
{
    public override void OnActionReceived(ActionBuffers actions) 	/// Override because it already exists in the Agent class
    {
        Debug.Log(actions.DiscreteActions[0]); 						/// Let's print the first (and only) action on our agent
    }
}
```

### C. Training an agent that performs discrete actions

1. In the cmd prompt, in the `UnityMLTutorial` project's python venv that you created during installation, run `mlagents-learn`. You should see a message appear saying `Listening on port 5004. Start training by pressing the Play button in the Unity Editor.`.
2. Press the play button in the Unity editor 
 - Note the training data printed in the cmd prompt
 - See the 5 actions being printed to the Unity console (0-4, because we specified in the Behavior Parameters component that `actions.DiscreteActions[0]` has a branch size of 5)
3. Stop playing in the Unity editor and notice that the output on the cmd prompt states that the results of the training were saved to a ONNX file:

```
INFO [subprocess_env_manager.py:220] UnityEnvironment worker 0: environment stopping.
INFO [trainer_controller.py:188] Learning was interrupted. Please wait while the graph is generated.
INFO [model_serialization.py:183] Converting to results\ppo\MoveToGoal\MoveToGoal-6080.onnx
INFO [model_serialization.py:195] Exported results\ppo\MoveToGoal\MoveToGoal-6080.onnx
INFO [torch_model_saver.py:143] Copied results\ppo\MoveToGoal\MoveToGoal-6080.onnx to results\ppo\MoveToGoal.onnx.
INFO [trainer_controller.py:81] Saved Model
```

### D. Training an agent that performs continuous actions

1. In the Behavior Parameters component, change the following params:
 - Vector Action: Space Type: `Continuous`
 - Vector Action: Space Size: `1`
2. In the "MoveToGoalAgent" script, log `actions.ContinuousActions[0]` instead of `actions.DiscreteActions[0]`:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class MoveToGoalAgent : Agent
{
    public override void OnActionReceived(ActionBuffers actions)
    {
        Debug.Log(actions.ContinuousActions[0]);
    }
}
```
3. In the cmd prompt run `mlagents-learn` again but this time you'll have to either append `--force` to overwrite the previous tranining on the default ID, or assign a new ID by appending `--run-id=<NEW_ID_NAME>` (i.e., `mlagents-learn --run-id=Test2`)
4. Press Play in the Unity editor
 - Note what continuous actions look like in the Unity editor console: -1 through 1

### E. Collecting observations

1. In order to collect observations our agent needs a **sensor** in order to observe its environment. Observations are the inputs for the AI. Add the `CollectObservations()` Agent function to our "MoveGoalToAgent" script:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors; /// Add the Sensors namespace

public class MoveToGoalAgent : Agent
{
    public override void CollectObservations(VectorSensor sensor) 	/// Add the CollectObservations function
    {
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        Debug.Log(actions.ContinuousActions[0]);
    }
}
```

2. What data does our agent need in order to solve the problem we're giving it?
 - Problem: teach the agent to move toward the goal and not fall off the platform
 - Data needed: agent position, goal position

Pass this data into the AI using the `VectorSensor`'s `AddObservation()` function:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform; 			/// Get "Goal" gameobject transform by adding it to an inspector field

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition); 			/// Add agent position as input
        sensor.AddObservation(targetTransform.localPosition);		/// Add goal position as input
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        Debug.Log(actions.ContinuousActions[0]);
    }
}
```

3. In the Behavior Parameters component, set the following params:
 - Vector Observation: Space Size: `6`; each position we're passing in is 3 floats, so 3 floats times 2 positions total is 6
 - Vector Observation: Stacked Vectors: `1`; if this is 1, it will only take into account this current observation when making a decision; if 2, it will take into account this observation *and* the previous observation when making this decision
 - Vector Action: Space Size: `2`; 1 float for the x pos of the agent, and 1 float for the z pos of the agent; movement in these directions are the only actions our agent takes

4. Edit the `OnActionReceived()` function in our agent script to receive these x and z movements and apply these values to our agent's position:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform;

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTransform.localPosition);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];										/// Get first action idx and assign to x
        float moveZ = actions.ContinuousActions[1];										/// Get second action idx and assign to z

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 3f;	/// Set agent pos based on these values
    }
}
```

### F. Rewarding agent for correct behavior

There are 2 ways to assign awards to an agent:
 - `AddReward(float increment)`: increment an overall reward amount (i.e., agent reaches a checkpoint toward the goal)
 - `SetReward(float reward)`: set the single reward (i.e., agent reaches the goal itself)

1. Call SetReward on the Agent gameobject when it collides with Goal gameobject collider, then, once the reward has been received by the agent, end the training **episode** ("one sequence of states, actions and rewards, which ends with terminal state. For example, playing an entire game can be considered as one episode, the terminal state being reached when one player loses/wins/draws." [src](https://stats.stackexchange.com/a/250955)):

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform;

    public override void OnEpisodeBegin()       		/// Reset the scene to train again every time the training episode begins
    {
        transform.localPosition = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTransform.localPosition);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 3f;
    }

    private void OnTriggerEnter(Collider other)       	/// Triggered when our Agent gameobject's collider enters another gameobject's collider
    {
        if (other.TryGetComponent<Goal>(out Goal goal))	/// If we hit a Goal
        {
            SetReward(1f);                              /// The input val here only matters relative to other rewards
            EndEpisode();
        }
    }
}
```

2. Set a penalty on the Agent gameobject when it collides with a Wall, then also end the episode:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform;

    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTransform.localPosition);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 3f;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out Wall wall))  	/// If we hit a Wall
        {
            SetReward(-1f);                             	/// Set penalty
            EndEpisode();
        }
    }
}
```

### G. Testing before training

To test we want to be able to send actions ourselves. In order to send actions ourselves we use the `Heuristic(in ActionBuffers actionsOut)` function from the Agent namespace.

A **heuristic** can be understood as a way of identifying attributes that helps the agent understand something (i.e., identifying pedal length might be a heuristic an agent uses to classify flowers in an image)

1. Use `Heuristic()` to modify the actions that will be received by the `OnActionReceived()` function:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform;

    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTransform.localPosition);
    }

    public override void Heuristic(in ActionBuffers actionsOut)                 /// Assign custom heuristics to the actions for this agent for testing
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;  /// Instantiate some actions (continuous actions in this case)
        continuousActions[0] = Input.GetAxisRaw("Horizontal");                  /// Set `moveX` with L, R arrow keys (see Edit > Project Settings > Input Manager)
        continuousActions[1] = Input.GetAxisRaw("Vertical");                    /// Set `moveY` with UP, DN arrow keys
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 3f;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }
}
```

2. In the Unity editor, in the Agent object's Behavior Parameters component, set the following params:
 - Behavior Type: `Heuristic Only`; if you don't have python w ml-agents running and no model selected `Default` will automatically use heuristics anyway

3. Play the scene and test hitting a Wall and a Goal by controlling your Agent with the arrow keys, making sure the episode ends and everything is reset properly.

### H. Training

1. In the Unity editor, in the Agent object's Behavior Parameters component, set the following param:
 - Behavior Type: `Default`; we are going to be running python with ml-agents now so heuristics won't automatically be used

2. In the Agent object's MoveTo Goal Agent component, set the following param:
 - Max Step: `5000`; after 5000 steps, end the episode and restart to keep the Agent's pos from stalling forever during early stages of training 

3. In the cmd prompt, in your venv, run `mlagents-learn --run-id=Test3`, then go press play in the Unity editor to begin training

Notice that it takes forever for the agent to touch the Goal for the first time, and often becomes somewhat stagnant in one area. Training can be sped up by increasing the number of agents being trained simultaneously. 

4. Create an empty gameobject named "Environment" and put your entire scene except for the Main Camera and Direction Light objects into it, then dupe the Environment object as many times as you want (anywhere around 10-20 total agents should be quick enough)

5. If you want to be able to better visualize when an agent wins or loses, pass the WinMat and LoseMat materials as inspector params into the MoveToGoalAgent.cs script and assign them to the Platform when the agent hits a Wall or the Goal:

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToGoalAgent : Agent
{
    [SerializeField] private Transform targetTransform;
    [SerializeField] private Material winMat;                   /// Pass in win material, assign WinMat material to this param
    [SerializeField] private Material loseMat;                  /// Pass in lose material, assign LoseMat material to this param
    [SerializeField] private MeshRenderer floorMeshRenderer;    /// Pass in platform renderer, assign the Goal object's Mesh Renderer to this param

    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(targetTransform.localPosition);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal") * 3f;
        continuousActions[1] = Input.GetAxisRaw("Vertical") * 3f;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        transform.localPosition += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 3f;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(+1f);
            floorMeshRenderer.material = winMat;                /// Assign win material
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-1f);
            floorMeshRenderer.material = loseMat;               /// Assign lose material
            EndEpisode();
        }
    }
}
```

6. Run `mlagents-learn` again but with an official name, i.e., `mlagents-learn --run-id=MoveToGoal` and press play in the Unity editor. Wait until all agents consistently succeed and failures are nonexistent.

When finished training press stop in the Unity editor and check the cmd prompt for the .onnx file location. This file is our neural network model, aka our "brain," which is to be used during inference to apply the results to this training.

III. Using a trained model
--------------------------

1. Copy the MoveToGoal.onnx file that was generated in the final training round and paste it into your project's Assets folder.
2. Disable all Environment objects so that you're left with one again
3. In the Agent object's Behvaior Parameters component, set the following params:
 - Model: drag in the MoveToGoal.onnx file
 - Behavior Type: `Default` or `Inference Only`; now that there's a Model, `Default` will use this model even without training
4. Press play; the Agent will use the MoveToGoal model to successfully move to Goal

IV. Credits
-------

Based on: https://www.youtube.com/watch?v=zPFU30tbyKs
