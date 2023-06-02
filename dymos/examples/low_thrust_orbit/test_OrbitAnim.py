from OrbitAnim import OrbitAnim

orbit = OrbitAnim('orbital_elements_min_p.txt')
orbit.set_elev(30)
orbit.set_azim(-120)
orbit.run_animation()