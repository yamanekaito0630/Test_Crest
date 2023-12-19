import cv2
import numpy as np
from pyvirtualdisplay import Display
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# d = Display()
# d.start()

# Unity環境の生成
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1, width=1500, height=600, capture_frame_rate=50)
env = UnityEnvironment('apps/UnderWaterDrones_IM_Round_10Robots_At7_V1', side_channels=[channel], no_graphics=False, worker_id=50)

# Unity環境のリセット
env.reset()
env.reset()

behavior_names = list(env.behavior_specs.keys())
print('behavior_names:', behavior_names)

# BehaviorSpecの取得
behavior_spec = env.behavior_specs[behavior_names[0]]

# BehaviorSpecの情報確認
print('\n== BehaviorSpecの情報の確認 ==')
print('\nobservation_spec:', behavior_spec.observation_specs)
print('\naction_spec:', behavior_spec.action_spec)

# 動作確認
while True:
    # 現在のステップの情報の取得
    decision_steps, terminal_steps = env.get_steps(behavior_names[0])

    # # DecisionStepsの情報の確認
    # print('\n== DecisionStepsの情報の確認 ==')
    # print('\nobs1:', decision_steps.obs[0][0]*255)
    # print('\nobs1:', decision_steps.obs[0][0][3::4])

    print('\nobs2:', decision_steps.obs[1][0])
    # print('\nobs3:', decision_steps.obs[2][1])
    # print('\nobs4:', decision_steps.obs[3][1][4:])
    # print('\nreward:', decision_steps.reward)
    # print('\nagent_id:', decision_steps.agent_id)
    # print('\naction_mask:', decision_steps.action_mask)

    # TerminalStepsの情報の確認
    # print('\n== TerminalStepsの情報の確認 ==')
    # print('\nobs:', terminal_steps.obs)
    # print('\nreward:', terminal_steps.reward)
    # print('\nagent_id', terminal_steps.agent_id)
    # print('\ninterrupted', terminal_steps.interrupted)

    # 行動の決定
    # print('len robo:',len(decision_steps.agent_id))
    for i in decision_steps.agent_id:
        img = decision_steps.obs[0][0] * 255
        simple_img = cv2.resize(img, (400, 400))[:, :, ::-1]
        #cv2.imwrite('test2_img.png', simple_img)

        action = (np.random.rand(3) * 2.0 - 1.0).reshape(1, 3).astype(np.float32)
        # print('action:', action)
        action_tuple = ActionTuple(continuous=action)
        env.set_action_for_agent(behavior_names[0], i, action_tuple)

    # Unity環境の1ステップ実行
    env.step()
    # print('step!')

    # with open('test.txt', mode="a") as f:
    # print('test')
    # DecisionStepsの情報の確認
    # print('\n== DecisionStepsの情報の確認 ==')
    # print('\nobs:' + str(decision_steps.obs))
    # print('\nreward:' + str(decision_steps.reward))
    # print('\nagent_id:' + str(decision_steps.agent_id))
    # print('\naction_mask:' + str(decision_steps.action_mask))

    # TerminalStepsの情報の確認
    # print('\n== TerminalStepsの情報の確認 ==')
    if len(terminal_steps.interrupted) > 0:
        env.reset()
        print('\nobs:' + str(terminal_steps.obs[1][0]))
        # print('\nreward:' + str(terminal_steps.reward))
        print('\nagent_id' + str(terminal_steps.agent_id))
        # print('\ninterrupted' + str(terminal_steps.interrupted))

# Unity環境のクローズ
env.close()
