deploy:
	python3 -m build
	sudo docker build . -t wheel_img
	sudo docker cp $(sudo docker create --rm wheel_img):/tmp/fastpivot-0.1.11-cp37-cp37m-manylinux_2_24_x86_64.whl wheelhouse/
	mv wheelhouse/* dist
	rm dist/fastpivot-*-cp37-cp37m-linux_x86_64.whl
	python3 -m twine upload --skip-existing --repository pypi dist/*

clean:
	rm -r build
	rm -r dist
	rm -r fastpivot.egg-info
	rm ./fastpivot/pivot.cpython-37m-x86_64-linux-gnu.so
	rm -r ./fastpivot/__pycache__
	rm ./fastpivot/pivot.cpp

devbuild:
	python3 setup.py build_ext --inplace

devclean:
	rm ./fastpivot/pivot.cpython-37m-x86_64-linux-gnu.so
	rm -r ./fastpivot/__pycache__
	rm ./fastpivot/pivot.cpp