#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
namespace
{
std::vector<float> latest_keyboard_velocity_command = {0.0f, 0.0f, 0.0f};
}

// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {0.5f, 0.0f, 0.0f}},
        {"s", {-0.3f, 0.0f, 0.0f}},
        {"a", {0.0f, 0.2f, 0.0f}},
        {"d", {0.0f, -0.2f, 0.0f}},
        {"q", {0.0f, 0.0f, 0.1f}},
        {"e", {0.0f, 0.0f, -0.1f}}
    };
    if (key_commands.find(key) != key_commands.end())
    {
        latest_keyboard_velocity_command = key_commands[key];
    }
    else if (key == "x")
    {
        latest_keyboard_velocity_command = {0.0f, 0.0f, 0.0f};
    }
    return latest_keyboard_velocity_command;
}

// Match the MJLab training observation: phase follows global episode time,
// but is masked to [0, 0] while the commanded velocity norm is below 0.1.
REGISTER_OBSERVATION(command_conditioned_gait_phase)
{
    const float period = params["period"].as<float>();
    const float stand_command_threshold =
        params["stand_command_threshold"].as<float>(0.1f);

    const float elapsed = static_cast<float>(env->episode_length) * env->step_dt;
    const float global_phase = std::fmod(elapsed, period) / period;

    float command_norm_squared = 0.0f;
    for (const float value : latest_keyboard_velocity_command)
    {
        command_norm_squared += value * value;
    }
    if (std::sqrt(command_norm_squared) < stand_command_threshold)
    {
        return {0.0f, 0.0f};
    }

    return {
        std::sin(global_phase * 2.0f * static_cast<float>(M_PI)),
        std::cos(global_phase * 2.0f * static_cast<float>(M_PI))
    };
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            []()->bool{ return keyboard->on_pressed && keyboard->key() == "0"; },
            FSMStringMap.right.at("Passive")
        )
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
