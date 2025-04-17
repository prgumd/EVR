###############################################################################
#
# A simple PID loop
#
# History:
# Spring 24 - Jack Mirenzi, Levi Burner - Created class
# 06-27-24 - Levi Burner Split SimplePID into standalone algorithm
#
###############################################################################

class SimplePID:
  def __init__(self,k_p=1.,k_i=0.,k_d=0., dt=None, bounds=None,tau=0.1, tau_in=None, tau_setpoint=None):
    self.k_p=k_p; self.k_i=k_i; self.k_d=k_d;self.bd=bounds
    self.dt=dt
    self.goal = 0.0; self.response=0.0
    self.deriv_filter=0.0; self.integral=0.0
    self.error_filter=0.0
    self.setpoint_filter=0.0
    self.tau=tau
    self.tau_in=tau_in
    self.tau_setpoint=tau_setpoint
  def set_goal(self,goal):
    self.goal=goal
  def update(self,value,dt=None):
    if dt is not None:
      self.dt = dt

    if self.tau_setpoint is not None:
      self.setpoint_filter += (self.dt/self.tau_setpoint)*(self.goal - self.setpoint_filter)
    else:
      self.setpoint_filter = self.goal

    if self.tau_in is not None:
      self.error_filter += (self.dt/self.tau_in)*((self.setpoint_filter - value) - self.error_filter)
      error = self.error_filter
    else:
      error = self.setpoint_filter - value

    last_deriv_filter = self.deriv_filter
    self.deriv_filter += (self.dt/self.tau)*(error - self.deriv_filter)
    deriv = (self.deriv_filter - last_deriv_filter) / self.dt

    self.integral += self.dt*error
    self.response = self.k_p*(error + self.k_i*self.integral + self.k_d*deriv)
    if self.bd is not None:
      self.response=max(self.bd*-1,min(self.bd,self.response))
    return self.response
  def reset(self):
    self.deriv_filter=0.0;self.integral=0.0
  def __str__(self) -> str:
    return f"({self.deriv_filter:.3f}, {self.goal:.3f}, {self.response:.3f}) "
