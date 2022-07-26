<?php

class Images extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();

        // load model
        $this->load->model('images_model');

        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));

        // load library
        $this->load->library(array('session', 'form_validation'));

        // autentifikasi
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') == 3) {
            redirect("auth/logout");
        }
    }

    public function index()
    {
        $this->list();
    }

    public function list()
    {
        $data['title'] = 'Data Foto Nomor Plat';
        $data['subtitle'] = 'List Foto Nomor Plat';
        $data['role_list'] = '3';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('images/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('images/script', $data);
    }

    public function ajax_list($role = false)
    {
        $list = $this->images_model->get_datatables($role);
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $image) {
            $no++;
            $row = array();

            $row[] = '<input type="checkbox" class="data-check" value="' . $image->id . '" onclick="showBottomDelete()"/>';
            $row[] = '
            <a class="btn btn-sm btn-danger" href="javascript:void()" title="Hapus Data" onclick="deleteData(' . "'" . $image->id . "'" . ')"><i class="glyphicon glyphicon-trash"></i></a>
            ';
            $row[] = $image->name;
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->images_model->count_all($role),
            "recordsFiltered" => $this->images_model->count_filtered($role),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }

    public function tambah()
    {

        $this->load->model('images_model');

        if ($this->input->method() === 'post') {
            // the user id contain dot, so we must remove it
            $file_name = str_replace('.', '', '1');
            $config['upload_path']          = FCPATH . '/scanner/img/';
            $config['allowed_types']        = 'jpeg';
            $config['file_name']            = $file_name;
            $config['overwrite']            = true;
            $config['max_size']             = 1024; // 1MB
            $config['max_width']            = 1080;
            $config['max_height']           = 1080;

            $this->load->library('upload', $config);

            if (!$this->upload->do_upload('plat')) {
                $data['error'] = $this->upload->display_errors();
            } else {
                $uploaded_data = $this->upload->data();
                header('Location: ' . base_url() . 'scanner/test.php');

                $this->session->set_flashdata('message', 'plat updated!');
                // redirect(site_url('admin/setting'));
            }
        }

        // $this->load->view('admin/setting_upload_avatar.php', $data);
        $data['title'] = 'Foto Nomor Plat';
        $data['subtitle'] = 'Tambah Foto Nomor Plat';
        $data['role_list'] = '3';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('images/tambah', $data);
        $this->load->view('templates/admin-footer');
        // $this->load->view('images/script', $data);
    }

    public function keluar()
    {

        $this->load->model('images_model');

        if ($this->input->method() === 'post') {
            // the user id contain dot, so we must remove it
            $file_name = str_replace('.', '', '2');
            $config['upload_path']          = FCPATH . '/scanner/img/';
            $config['allowed_types']        = 'jpeg';
            $config['file_name']            = $file_name;
            $config['overwrite']            = true;
            $config['max_size']             = 1024; // 1MB
            $config['max_width']            = 1080;
            $config['max_height']           = 1080;

            $this->load->library('upload', $config);

            if (!$this->upload->do_upload('plat')) {
                $data['error'] = $this->upload->display_errors();
            } else {
                $uploaded_data = $this->upload->data();
                header('Location: ' . base_url() . 'scanner/test2.php');

                $this->session->set_flashdata('message', 'plat updated!');
                // redirect(site_url('admin/setting'));
            }
        }

        // $this->load->view('admin/setting_upload_avatar.php', $data);
        $data['title'] = 'Foto Nomor Plat  Keluar';
        $data['subtitle'] = 'Tambah Foto Nomor Plat Keluar';
        $data['role_list'] = '3';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('images/tambah', $data);
        $this->load->view('templates/admin-footer');
        // $this->load->view('images/script', $data);
    }


    public function ajax_delete($id)
    {
        // cari data id di database
        $getData_id = $this->user_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $name = $getData_id->fullname;
        $email = $getData_id->email;

        // buat txt pesan
        $txt = "Menghapus data " . $name . " (" . $email . ")";

        // masukan dalam array
        $data_log = array(
            'alert' => 'danger',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $this->user_model->delete_by_id($id);
        echo json_encode(array("status" => TRUE));
    }
}
