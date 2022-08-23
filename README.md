# Lidar problem

## Problema 1

Debido a que bajamos el lidar a 5cm de la base, ahora detecta las ruedas y los servos. por lo que el angulo se debe restringir. Aunque creo que si esta muy cerca el lidar lo omite por lo que no deberia ser un problema. De todas formas, para ajustar el angulo probe 2 soluciones.

### Solucion 1
Script que hace 0 los rango de los angulos que no se necesitan:
KSAM-2022-Robotic-Competition/ros/src/turtlebot3_bringup/scripts/lidar_corr.py

No funciono

### Solucion 2
Usar laser filter, se agrega en los launchers de karto, gmapping slams y en los de navegacions:
```
  <node pkg="laser_filters" type="scan_to_scan_filter_chain" output="screen" name="laser_filter">
    <rosparam file="$(find robot_slam)/config/angle_filter.yaml" command="load"/>
  </node>
```
Funciona, pero al usar los scripts de navegacion queda una sombra atras del robot (en la simulacion) y no anda bien.
Tambien se cambio el urdf para que el sensor funcione en 360 grados.
 KSAM-2022-Robotic-Competition/ros/src/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro 
```
          <horizontal>
            <samples>360</samples>          <!-- 260  without filter-->
            <resolution>1</resolution>
            <min_angle>0.0</min_angle>      <!-- -2.26893  without filter-->
            <max_angle>6.28319</max_angle>  <!-- 2.26893  without filter-->
          </horizontal>
```
Foto de la sombra.


## Problema 2
Al usar karto con el robot real las murallas se desplazan y no mapea bien, posibles problema odom, imu? o error del lidar.

Foto del error del lidar
