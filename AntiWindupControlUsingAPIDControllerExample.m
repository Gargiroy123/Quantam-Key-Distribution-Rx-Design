%% Anti-Windup Control Using a PID Controller

% Copyright 2009-2020 The MathWorks, Inc.

%% Model Description
% This example shows how to use anti-windup schemes to prevent integration
% wind-up in PID controllers when the actuators are saturated. We use the
% PID Controller block in Simulink(R) which features two built-in
% anti-windup methods, |back-calculation| and |clamping|, as well as a
% tracking mode to handle more complex scenarios.

%%
% The plant to be controlled is a saturated first-order process with
% dead-time.
%
% We start by opening the model.

open_system('sldemo_antiwindup')

%%
% *Figure 1:* Simulink model of PID control of a plant with input
% saturation.

%%
% To open this model, type |sldemo_antiwindup| in a MATLAB(R) terminal.

%%
% The PID Controller has been tuned with saturation ignored using the
% PID tuner of Simulink(R) Control Design(TM).

%%
% The controlled plant is a first-order process with dead-time described
% by
%
% $$P(s)=\frac{1}{10s+1}e^{-2s}$$
% 
% The plant has known input saturation limits of [-10, 10], which are
% accounted for in the Saturation block labeled Plant Actuator. The PID 
% Controller block in Simulink features two built-in anti-windup methods
% that allow the PID Controller block to account for the available
% information about the plant input saturation.

%% Performance Without Using Anti-Windup
% First, we examine the effect of saturation on the closed-loop when the
% saturation model is not considered by the PID Controller block.
% Simulating the model in Figure 1 generates the results shown below.

open_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1'); % This ensures legends get displayed
simout1= sim('sldemo_antiwindup','ReturnWorkspaceOutputs','on');
close_system('sldemo_antiwindup/Scope1');

%%
% *Figure 2:* Setpoint vs. measured output with no anti-windup.

%%

close_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1');

%%
% *Figure 3:* Controller output and saturated input with no anti-windup.

%%
% Figures 2 and 3 highlight two of the problems that arise when controlling
% a system with input saturation:
%
% # When the setpoint value is 10, the PID control signal reaches a
% steady-state at about 24, outside the range of the actuator. The
% controller is therefore operating in a nonlinear region where increasing
% the control signal has no effect on the system output, a condition known
% as _winding up_. Note that the dc-gain of the plant is unity, and
% therefore there is no reason for the controller output to have a
% steady-state value outside the actuator's range.
% # When the setpoint value becomes 5, there is a considerable delay before
% the PID controller output returns to within the actuator range.
%
% Designing the PID controller to account for the effect of saturation will
% improve its performance by allowing it to operate in the linear region
% most of the time and recover quickly from nonlinearity. Anti-windup
% circuitry is one way to achieve this.

%% Configuring the Block for Anti-Windup Based on Back-Calculation
% The back-calculation anti-windup method uses a feedback loop to discharge
% the PID Controller internal integrator when the controller hits specified
% saturation limits and enters nonlinear operation. To enable anti-windup,
% go to the *Output Saturation* tab in the block dialog; select *Limit
% output*; and enter the plant's saturation limits. Then, select
% *back-calculation* from the *Anti-windup method* menu and specify the
% *Back-calculation coefficient (Kb)*. The inverse of this gain is the time
% constant of the anti-windup loop. In this example, the back-calculation
% gain is chosen to be 1. For more information on how to choose this value,
% see Reference [1].

%%
% 
% <<../blkDlgAntiWindup.png>>
%
% *Figure 4:* Enabling the back-calculation anti-windup method.

set_param('sldemo_antiwindup/PID Controller','LimitOutput','on',...
    'UpperSaturationLimit','10','LowerSaturationLimit','-10',...
    'AntiWindupMode','back-calculation');


%%
% Once back-calculation is enabled, the block has an internal tracking loop
% that discharges the Integrator output.

open_system('sldemo_antiwindup/PID Controller','force');

%%
% *Figure 5:* Under-mask view of the PID Controller block with
% back-calculation.

%%
% Figures 6 and 7 illustrate the result of simulating the model with
% anti-windup activated. Note how quickly the PID control signal returns to
% the linear region and how fast the loop recovers from saturation.

close_system('sldemo_antiwindup/PID Controller');
close_system('sldemo_antiwindup/Scope1');
open_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1'); % To force data to plot on scope
simout2= sim('sldemo_antiwindup','ReturnWorkspaceOutputs','on');
close_system('sldemo_antiwindup/Scope1');

%%
% *Figure 6:* Setpoint vs. measured output with back-calculation.

close_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1');

%%
% *Figure 7:* Controller output and saturated input with back-calculation.

%%
% Figure 7 shows that the controller output |u(t)| and the saturated input
% |SAT(u)| coincide with each other because *Limit output* is enabled.

%%
% To better visualize the effect of anti-windup, Figure 8 illustrates the
% plant measured output |y(t)| with and without anti-windup.

close_system('sldemo_antiwindup/Scope1');
t1 = get(simout1,'tout');  y1 = get(simout1,'yout');
t2 = get(simout2,'tout');  y2 = get(simout2,'yout');
figure('Tag','sldemo_antiwindup');
plot(t1,y1,t2,y2);axis([0 t1(end) round(min([y1;y2])-2) round(max([y1;y2])+2)]);
title('Measured output');
legend('Without anti-windup','With anti-windup');

%%
% *Figure 8:* Measured output with and without anti-windup.

%% Configuring the Block for Anti-Windup Based on Integrator Clamping
% Another commonly used anti-windup strategy is based on conditional
% integration. To enable anti-windup, go to the *PID Advanced* tab in the
% block's dialog; select *Limit output*; and enter the plant's saturation
% limits. Then, select *clamping* from the *Anti-windup method* menu.

open_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1'); % To force data to plot on scope
set_param('sldemo_antiwindup/PID Controller','AntiWindupMode','clamping');
sim('sldemo_antiwindup');
close_system('sldemo_antiwindup/Scope1');

%%
% *Figure 9:* Setpoint vs. measured output with clamping.

%%
close_system('sldemo_antiwindup/Scope');
open_system('sldemo_antiwindup/Scope1');

%%
bdclose('sldemo_antiwindup')
close(findobj('Type','figure','Tag','sldemo_antiwindup'))

%%
% *Figure 10:* Controller output and saturated input with clamping.

%%
% Figure 10 shows that the controller output |u(t)| and the saturated input
% |SAT(u)| coincide with each other because *Limit output* is enabled.


%%
% For more information on when to use _clamping_, see Reference [1].

%% Using Tracking Mode to Handle Complex Anti-Windup Scenarios
% The previously discussed anti-windup strategies relied on built-in
% methods to process the saturation information provided to the block via
% its dialog. For those built-in techniques to work as intended, two
% conditions must be met:
%
% # The plant's saturation limits are known and can be entered into the
% dialog of the block.
% # The PID Controller output signal is the only signal feeding the
% actuator.
%
% These conditions may be restrictive when handling general anti-windup
% scenarios. The PID Controller block features a tracking mode that
% allows the user to set up a back-calculation anti-windup loop externally.
% The following two examples are considered to
% illustrate the use of tracking mode for anti-windup purposes:
%
% # Anti-windup for saturated actuators with cascaded dynamics
% # Anti-windup for PID control with Feedforward


%% Constructing Anti-Windup Circuitry for Saturated Actuators with Cascaded Dynamics
% In the following model, the actuator has complex dynamics. This is common
% when an actuator has its own closed-loop dynamics. The PID controller is
% in an outer loop and sees the actuator dynamics as an inner loop, or
% simply a cascaded saturated dynamics as shown in Figure 1.

open_system('sldemo_antiwindupactuator');

%%
% *Figure 11:* Simulink model of PID controller with cascaded actuator
% dynamics.

%%
% To open this model, type |sldemo_antiwindupactuator| in a MATLAB
% terminal.

%%
% In this case, a successful anti-windup strategy requires feeding back the
% actuator output to the tracking port of the PID Controller block as shown
% in Figure 11. To configure the |tracking mode| of the PID Controller
% block, go to the *PID Advanced* tab in the block's dialog; select *Enable
% tracking mode*; and specify the gain |Kt|. The inverse of this gain is
% the time constant of the tracking loop. For more information on how to
% choose this gain, see Reference [1].

%%
% Figures 12 and 13 show that the plant's measured output |y(t)| and the
% controller output |u(t)| are responding almost immediately to changes in 
% the setpoint. Without the anti-windup circuit, these responses would be
% sluggish with long delays.

open_system('sldemo_antiwindupactuator/Scope');
open_system('sldemo_antiwindupactuator/Scope1'); % This ensures legends get displayed
sim('sldemo_antiwindupactuator');
close_system('sldemo_antiwindupactuator/Scope1');

%%
% *Figure 12:* Setpoint vs. measured output.

%%
close_system('sldemo_antiwindupactuator/Scope');
open_system('sldemo_antiwindupactuator/Scope1');

%%
% *Figure 13:* Controller output and effective saturated input.

%%
bdclose('sldemo_antiwindupactuator')

%% Constructing Anti-Windup Circuitry for PID Control with Feedforward
% In another common control configuration, the actuator receives a control
% signal that is a combination of a PID control signal and a feedforward
% control signal.

%%
% To accurately build a back-calculation anti-windup loop, the tracking
% signal should subtract the contribution of the feedforward signal. This
% allows the PID Controller block to know its share of the effective
% control signal applied to the actuator.

%%
% The following model includes a feedforward control.

open_system('sldemo_antiwindupfeedforward');

%%
% *Figure 14:* Simulink model of PID controller with feedforward, and plant
% input saturation.

%%
% The feedforward gain is selected to be unity here because the plant has a
% dc-gain of 1.

%%
% To open this model, type |sldemo_antiwindupfeedforward| in a MATLAB
% terminal.

%%
% Figures 15 and 16 show that the plant's measured output |y(t)| and the
% controller output |u(t)| are responding almost immediately to changes in
% the setpoint. When the setpoint value is 10, note how in Figure 16 the
% controller output |u(t)| reduces to be within the range of the actuator.

open_system('sldemo_antiwindupfeedforward/Scope');
open_system('sldemo_antiwindupfeedforward/Scope1'); % This ensures legends get displayed
sim('sldemo_antiwindupfeedforward');
close_system('sldemo_antiwindupfeedforward/Scope1');

%%
% *Figure 17:* Setpoint vs. measured output without anti-windup.

%%
close_system('sldemo_antiwindupfeedforward/Scope');
open_system('sldemo_antiwindupfeedforward/Scope1');

%%
% *Figure 18:* Controller output and saturated input with anti-windup.

%%
bdclose('sldemo_antiwindupfeedforward')

%% Summary
% The PID Controller block supports several features that allow it to
% handle controller windup issues under commonly encountered industrial
% scenarios.


%% References
% # K. &Aring;str&ouml;m, T. H&auml;gglund, _Advanced PID Control_, ISA, Research Triangle
% Park, NC, August 2005.
