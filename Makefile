deploy:
	python3 -m build
	sudo docker build . -t wheel_img
	sudo docker cp $(sudo docker create --rm wheel_img):/tmp/fastpivot-*-cp37-cp37m-manylinux_2_24_x86_64.whl wheelhouse/
	mv wheelhouse/* dist
	rm dist/fastpivot-*-cp37-cp37m-linux_x86_64.whl
	python3 -m twine upload --skip-existing --repository testpypi dist/*