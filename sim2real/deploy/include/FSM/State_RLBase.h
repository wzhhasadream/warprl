// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <iomanip>
#include <sstream>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Initialize the commanded target to the measured pose before the
        // policy thread starts. This prevents run() from briefly publishing
        // the ActionManager's zero-initialized processed target.
        const Eigen::VectorXf q_start_eigen = env->robot->data.joint_pos;
        std::vector<float> q_start(q_start_eigen.data(),
                                   q_start_eigen.data() + q_start_eigen.size());
        const auto action_cfg = env->cfg["actions"]["JointPositionAction"];
        const auto action_scale = action_cfg["scale"].as<std::vector<float>>();
        std::vector<float> startup_action(q_start.size(), 0.0f);
        for (std::size_t i = 0; i < startup_action.size(); ++i)
        {
            startup_action[i] =
                (q_start[i] - env->robot->data.default_joint_pos[i]) /
                action_scale[i];
        }
        env->action_manager->process_action(startup_action);

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this, q_start, action_scale]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Always perform a local safe-start transition. FixStand normally
            // approaches this pose too, but the user can enter Velocity before
            // its interpolation has completed.
            auto sleepTill = clock::now() + dt;
            constexpr float transition_duration = 2.0f;
            constexpr float hold_duration = 0.5f;
            const int transition_steps = static_cast<int>(
                transition_duration / env->step_dt);
            const int hold_steps = static_cast<int>(hold_duration / env->step_dt);

            spdlog::info(
                "Policy safe start: moving to default pose for {:.1f}s, holding for {:.1f}s",
                transition_duration, hold_duration);

            for (int startup_step = 1;
                 policy_thread_running && startup_step <= transition_steps + hold_steps;
                 ++startup_step)
            {
                float phase = std::min(
                    1.0f,
                    static_cast<float>(startup_step) /
                        static_cast<float>(transition_steps));
                // Smoothstep gives zero velocity at both ends of the move.
                phase = phase * phase * (3.0f - 2.0f * phase);

                std::vector<float> action(q_start.size(), 0.0f);
                for (std::size_t i = 0; i < action.size(); ++i)
                {
                    const float q_target =
                        (1.0f - phase) * q_start[i] +
                        phase * env->robot->data.default_joint_pos[i];
                    action[i] =
                        (q_target - env->robot->data.default_joint_pos[i]) /
                        action_scale[i];
                }
                env->action_manager->process_action(action);

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }

            if (!policy_thread_running) return;

            // Re-sample the settled state, fill every history slot from that
            // state, and clear last_action before the first ONNX inference.
            env->robot->update();
            env->reset();
            env->action_manager->process_action(
                std::vector<float>(q_start.size(), 0.0f));
            spdlog::info("Policy safe start complete; observation history reset, starting ONNX");

            while (policy_thread_running)
            {
                env->step();

                // Startup-only policy diagnostics: print every policy step for
                // the first 0.2 s, then at 10 Hz until 2.0 s.  Keeping this in
                // the policy thread gives a coherent raw-action/target/state
                // snapshot without racing the command publishing thread.
                const auto step = env->episode_length;
                if (step <= 10 || (step <= 100 && step % 5 == 0))
                {
                    const auto raw_action = env->action_manager->action();
                    const auto q_target = env->action_manager->processed_actions();
                    const auto &q_actual = env->robot->data.joint_pos;
                    std::vector<float> q_error(q_target.size(), 0.0f);
                    for (std::size_t i = 0; i < q_error.size(); ++i)
                    {
                        q_error[i] = q_target[i] - q_actual[i];
                    }

                    const auto format_vector = [](const auto &values)
                    {
                        std::ostringstream stream;
                        stream << std::fixed << std::setprecision(3) << "[";
                        for (std::size_t i = 0; i < values.size(); ++i)
                        {
                            if (i != 0) stream << ", ";
                            stream << values[i];
                        }
                        stream << "]";
                        return stream.str();
                    };

                    spdlog::info(
                        "Policy startup step={} time={:.3f}s raw_action={}",
                        step, step * env->step_dt, format_vector(raw_action));
                    spdlog::info("Policy startup q_target={}", format_vector(q_target));
                    spdlog::info("Policy startup q_actual={}", format_vector(q_actual));
                    spdlog::info("Policy startup q_error={}", format_vector(q_error));
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
