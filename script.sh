for i in {1..1000}
do
echo $i
python SantiagoBeltran_Ejercicio10.py $i >> DATOS.TXT
done
python graficar.py

